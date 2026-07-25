// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embedded WebAssembly decode runtime.
//!
//! [`WasmKernel`] wraps a compiled `wasmtime` module and drives the host/guest ABI. A kernel is
//! the portable decoder for one array encoding and receives the encoding's **real serialized
//! parts** — the same `(len, metadata, buffers, children)` a native `VTable::deserialize` gets:
//!
//! 1. `vx_children` tells the host each serialized child's dtype and length (only the encoding
//!    knows them);
//! 2. the host decodes those children (natively or through other kernels), copies the node's raw
//!    buffers and the decoded children into guest memory, and calls `vx_decode` with everything in
//!    one frame;
//! 3. the guest returns its output as a Vortex buffer table, which the host turns back into an
//!    array (see [`crate::convert`]) — or as a gather over a child it only named.
//!
//! Kernels are untrusted file data. The runtime is `wasmtime` with its default Cranelift backend
//! (not Winch/Pulley, which are less battle-tested); each decode runs in a fresh [`Store`] whose
//! linear memory growth is capped via [`StoreLimits`]. CPU-time bounding (fuel / epoch
//! interruption) is a planned follow-up — see `docs/design/wasm-encodings.md`.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use wasmtime::Caller;
use wasmtime::Engine;
use wasmtime::Extern;
use wasmtime::Instance;
use wasmtime::Linker;
use wasmtime::Memory;
use wasmtime::Module;
use wasmtime::ResourceLimiter;
use wasmtime::Store;
use wasmtime::StoreLimits;
use wasmtime::StoreLimitsBuilder;
use wasmtime::TypedFunc;

use crate::abi::ABI_VERSION;
use crate::abi::ABI_VERSION_EXPORT;
use crate::abi::ALLOC_EXPORT;
use crate::abi::CHILDREN_EXPORT;
use crate::abi::DECODE_EXPORT;
use crate::abi::HOST_LOG_IMPORT;
use crate::abi::HOST_MODULE;
use crate::abi::MEMORY_EXPORT;
use crate::convert::ArrayDescriptor;
use crate::convert::CHILD_ENTRY_SIZE;
use crate::convert::GuestMem;
use crate::convert::write_child;

/// Maximum linear memory a kernel may grow to in a single decode, as a coarse DoS guard against
/// untrusted kernels. Generous enough for legitimate decodes (wasm32 memory tops out at 4 GiB); a
/// starting value, not a tuned one.
const MAX_GUEST_MEMORY_BYTES: usize = 1 << 30;

/// Cap on the number of children a kernel may declare.
const MAX_CHILDREN: usize = 4096;

/// Frame flag bit 0: the parent dtype is nullable.
const FLAG_NULLABLE: u32 = 1;
/// Frame flags bits 8-15: the parent dtype's kind (0 other, 1 primitive, 2 bool, 3 utf8).
const PARENT_KIND_SHIFT: u32 = 8;
/// Frame flags bits 16-23: the parent's `PType` prost discriminant (when primitive).
const PARENT_PTYPE_SHIFT: u32 = 16;

/// Encode the frame flags word for a node dtype.
fn frame_flags(dtype: &DType) -> u32 {
    let mut flags = if dtype.is_nullable() {
        FLAG_NULLABLE
    } else {
        0
    };
    match dtype {
        DType::Primitive(ptype, _) => {
            flags |= 1 << PARENT_KIND_SHIFT;
            flags |= (*ptype as u32) << PARENT_PTYPE_SHIFT;
        }
        DType::Bool(_) => flags |= 2 << PARENT_KIND_SHIFT,
        DType::Utf8(_) => flags |= 3 << PARENT_KIND_SHIFT,
        _ => {}
    }
    flags
}

/// Child descriptor tags (see the guest SDK's `abi::child_descriptor`).
const TAG_PARENT: u8 = 0;
const TAG_PRIMITIVE: u8 = 1;
const TAG_BOOL: u8 = 2;
const TAG_UTF8: u8 = 3;
const DESCRIPTOR_SIZE: usize = 16;
const MODE_REFERENCE: u8 = 1;

/// Result frame tags (see the guest SDK's `abi::decode_result`).
const RESULT_MATERIALIZED: u32 = 0;
const RESULT_TAKE: u32 = 1;

/// How the kernel intends to use a serialized child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildMode {
    /// The kernel reads this child's element bytes: the host canonicalizes it and copies it into
    /// guest memory.
    Values,
    /// The kernel only names this child in its result: the host resolves it lazily, in its own
    /// encoding, and never canonicalizes or copies it.
    Reference,
}

/// The dtype, length, and access mode of one serialized child, as declared by the kernel.
#[derive(Debug, Clone)]
pub struct ChildDescriptor {
    /// The child's dtype.
    pub dtype: DType,
    /// The child's logical element count.
    pub len: usize,
    /// Whether the kernel reads this child or merely references it.
    pub mode: ChildMode,
}

/// What a kernel produced for a node.
pub enum KernelOutput {
    /// The kernel materialized the decoded array itself.
    Materialized(ArrayRef),
    /// The output is the child at `values_slot` gathered by `indices`. The host performs the
    /// gather, so the gathered child never crosses the sandbox boundary and may have any dtype.
    Take {
        /// The serialized child slot to gather from.
        values_slot: usize,
        /// One unsigned index per output element.
        indices: ArrayRef,
    },
}

/// Store state for a single decode: only the resource limiter (the v2 ABI has no host callbacks
/// that carry state).
struct HostState {
    limits: StoreLimits,
}

/// A compiled, reusable WebAssembly decoder kernel.
///
/// Compilation (the expensive step) happens once in [`WasmKernel::new`]. Each
/// [`decoder`](WasmKernel::decoder) call instantiates a fresh store and memory so that node
/// decodes are independent.
pub struct WasmKernel {
    engine: Engine,
    module: Module,
}

/// A live instance of a kernel for one decode call.
struct KernelInstance {
    store: Store<HostState>,
    memory: Memory,
    abi_version: TypedFunc<(), i32>,
    alloc: TypedFunc<i32, i32>,
    children: TypedFunc<(i32, i32), i32>,
    decode: TypedFunc<(i32, i32), i32>,
}

impl WasmKernel {
    /// Compile a kernel from raw `.wasm` bytes.
    ///
    /// Instantiates the module once to read its `vx_abi_version` export, so a kernel built against
    /// a different ABI is rejected here rather than misreading frames at decode time.
    pub fn new(wasm_bytes: impl AsRef<[u8]>) -> VortexResult<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes.as_ref())
            .map_err(|e| vortex_err!("failed to compile wasm kernel: {e}"))?;
        let kernel = Self { engine, module };

        let abi_version = kernel.read_abi_version()?;
        if abi_version != ABI_VERSION {
            vortex_bail!(
                "wasm kernel implements ABI version {abi_version}, but this host implements {ABI_VERSION}"
            );
        }
        Ok(kernel)
    }

    /// The ABI version the kernel declares via its `vx_abi_version` export.
    ///
    /// Always equal to [`ABI_VERSION`] for a kernel that compiled successfully; exposed so a
    /// writer can record the version it is embedding into a file.
    pub fn abi_version(&self) -> u32 {
        ABI_VERSION
    }

    fn read_abi_version(&self) -> VortexResult<u32> {
        let mut instance = self.instantiate()?;
        let version = instance
            .abi_version
            .call(&mut instance.store, ())
            .map_err(map_trap)?;
        u32::try_from(version)
            .map_err(|_| vortex_err!("wasm kernel reported a negative ABI version {version}"))
    }

    fn instantiate(&self) -> VortexResult<KernelInstance> {
        let mut store = Store::new(
            &self.engine,
            HostState {
                limits: StoreLimitsBuilder::new()
                    .memory_size(MAX_GUEST_MEMORY_BYTES)
                    .build(),
            },
        );
        store.limiter(|state| &mut state.limits as &mut dyn ResourceLimiter);

        let mut linker = Linker::<HostState>::new(&self.engine);
        linker
            .func_wrap(
                HOST_MODULE,
                HOST_LOG_IMPORT,
                |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                    if let Some(mem) = caller
                        .get_export(MEMORY_EXPORT)
                        .and_then(Extern::into_memory)
                    {
                        let mut buf = vec![0u8; len.max(0) as usize];
                        if mem.read(&caller, ptr.max(0) as usize, &mut buf).is_ok()
                            && let Ok(s) = std::str::from_utf8(&buf)
                        {
                            eprintln!("[wasm kernel] {s}");
                        }
                    }
                },
            )
            .map_err(|e| vortex_err!("failed to link {HOST_LOG_IMPORT}: {e}"))?;

        let instance: Instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| vortex_err!("failed to instantiate wasm kernel: {e}"))?;

        let memory = instance
            .get_memory(&mut store, MEMORY_EXPORT)
            .ok_or_else(|| vortex_err!("wasm kernel does not export memory '{MEMORY_EXPORT}'"))?;
        let abi_version = instance
            .get_typed_func::<(), i32>(&mut store, ABI_VERSION_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {ABI_VERSION_EXPORT}: {e}"))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, ALLOC_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {ALLOC_EXPORT}: {e}"))?;
        let children = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, CHILDREN_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {CHILDREN_EXPORT}: {e}"))?;
        let decode = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, DECODE_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {DECODE_EXPORT}: {e}"))?;

        Ok(KernelInstance {
            store,
            memory,
            abi_version,
            alloc,
            children,
            decode,
        })
    }

    /// Instantiate the kernel for one node decode.
    pub fn decoder(&self) -> VortexResult<WasmDecoder> {
        Ok(WasmDecoder {
            instance: self.instantiate()?,
        })
    }
}

/// A live kernel instance for decoding one serialized array node: first ask it which children the
/// node has ([`children`](Self::children)), decode them, then run [`decode`](Self::decode).
pub struct WasmDecoder {
    instance: KernelInstance,
}

impl WasmDecoder {
    /// Ask the kernel for the dtype and length of each of the node's `n_children` serialized
    /// children, given the encoding `metadata`.
    pub fn children(
        &mut self,
        dtype: &DType,
        len: usize,
        n_children: usize,
        metadata: &[u8],
    ) -> VortexResult<Vec<ChildDescriptor>> {
        let mut frame = Vec::with_capacity(20 + metadata.len());
        frame.extend_from_slice(&(len as u64).to_le_bytes());
        frame.extend_from_slice(&frame_flags(dtype).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(n_children)?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(metadata.len())?).to_le_bytes());
        frame.extend_from_slice(metadata);

        let frame_ptr = self.instance.upload(&frame)?;
        let result_ptr = self
            .instance
            .children
            .call(
                &mut self.instance.store,
                (frame_ptr as i32, i32::try_from(frame.len())?),
            )
            .map_err(map_trap)?;
        if result_ptr < 0 {
            vortex_bail!("wasm kernel {CHILDREN_EXPORT} returned error code {result_ptr}");
        }
        let descriptors = self.instance.read_descriptors(result_ptr as u32, dtype)?;
        if descriptors.len() != n_children {
            vortex_bail!(
                "wasm kernel declared {} children but the node has {n_children}",
                descriptors.len()
            );
        }
        Ok(descriptors)
    }

    /// Decode the node: `metadata` and `buffers` are its serialized parts, `children` the decoded
    /// [`ChildMode::Values`] child arrays (in declaration order — `Reference` children are absent,
    /// because they never enter guest memory).
    pub fn decode(
        &mut self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[ByteBuffer],
        children: &[Canonical],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<KernelOutput> {
        // Copy the decoded `Values` children into guest memory as fixed-size entries.
        let mut child_entries = Vec::with_capacity(children.len());
        for canonical in children {
            let mut guest = InstanceGuestMem {
                instance: &mut self.instance,
            };
            child_entries.push(write_child(canonical, ctx, &mut guest)?);
        }

        // Copy the raw buffers into guest memory.
        let mut buffer_entries = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let ptr = self.instance.upload(buffer.as_slice())?;
            buffer_entries.push((ptr, u32::try_from(buffer.len())?));
        }

        // Build the decode frame and run the kernel.
        let mut frame = Vec::with_capacity(
            24 + metadata.len() + buffers.len() * 8 + child_entries.len() * CHILD_ENTRY_SIZE,
        );
        frame.extend_from_slice(&(len as u64).to_le_bytes());
        frame.extend_from_slice(&frame_flags(dtype).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(metadata.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(buffers.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(child_entries.len())?).to_le_bytes());
        frame.extend_from_slice(metadata);
        for (ptr, buffer_len) in &buffer_entries {
            frame.extend_from_slice(&ptr.to_le_bytes());
            frame.extend_from_slice(&buffer_len.to_le_bytes());
        }
        for entry in &child_entries {
            frame.extend_from_slice(entry);
        }

        let frame_ptr = self.instance.upload(&frame)?;
        let result_ptr = self
            .instance
            .decode
            .call(
                &mut self.instance.store,
                (frame_ptr as i32, i32::try_from(frame.len())?),
            )
            .map_err(map_trap)?;
        if result_ptr < 0 {
            vortex_bail!("wasm kernel {DECODE_EXPORT} returned error code {result_ptr}");
        }

        // The result frame is [u32 tag] followed by a tag-specific body.
        let mut header = [0u8; 4];
        self.instance
            .memory
            .read(&self.instance.store, result_ptr as usize, &mut header)
            .map_err(|e| vortex_err!("failed to read result tag: {e}"))?;
        let tag = u32::from_le_bytes(header);

        let mem = self.instance.memory.data(&self.instance.store);

        match tag {
            RESULT_MATERIALIZED => {
                let (descriptor, _) = ArrayDescriptor::parse(mem, result_ptr as usize + 4)?;
                Ok(KernelOutput::Materialized(descriptor.build(mem, dtype)?))
            }
            RESULT_TAKE => {
                let bytes: [u8; 4] = mem
                    .get(result_ptr as usize + 4..result_ptr as usize + 8)
                    .and_then(|slice| slice.try_into().ok())
                    .ok_or_else(|| vortex_err!("truncated take result"))?;
                let slot = u32::from_le_bytes(bytes);
                let (descriptor, _) = ArrayDescriptor::parse(mem, result_ptr as usize + 8)?;
                Ok(KernelOutput::Take {
                    values_slot: slot as usize,
                    indices: descriptor.build_indices(mem)?,
                })
            }
            other => vortex_bail!("wasm kernel returned unknown result tag {other}"),
        }
    }
}

impl KernelInstance {
    /// Allocate guest memory via `vx_alloc` and copy `bytes` in, returning the guest offset.
    fn upload(&mut self, bytes: &[u8]) -> VortexResult<u32> {
        let ptr = self
            .alloc
            .call(&mut self.store, i32::try_from(bytes.len().max(1))?)
            .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} trapped: {e}"))?;
        let ptr = u32::try_from(ptr).map_err(|_| vortex_err!("guest returned bad pointer"))?;
        self.memory
            .write(&mut self.store, ptr as usize, bytes)
            .map_err(|e| vortex_err!("failed to write guest memory: {e}"))?;
        Ok(ptr)
    }

    /// Parse the `vx_children` result: `[u32 n][16-byte descriptors…]`.
    fn read_descriptors(&mut self, ptr: u32, parent: &DType) -> VortexResult<Vec<ChildDescriptor>> {
        let mut count = [0u8; 4];
        self.memory
            .read(&self.store, ptr as usize, &mut count)
            .map_err(|e| vortex_err!("failed to read child descriptors: {e}"))?;
        let n = u32::from_le_bytes(count) as usize;
        if n > MAX_CHILDREN {
            vortex_bail!("wasm kernel declared too many children: {n}");
        }

        let mut bytes = vec![0u8; n * DESCRIPTOR_SIZE];
        self.memory
            .read(&self.store, ptr as usize + 4, &mut bytes)
            .map_err(|e| vortex_err!("failed to read child descriptors: {e}"))?;

        (0..n)
            .map(|i| {
                let d = &bytes[i * DESCRIPTOR_SIZE..(i + 1) * DESCRIPTOR_SIZE];
                let nullability = Nullability::from(d[2] != 0);
                let dtype = match d[0] {
                    TAG_PARENT => parent.clone(),
                    TAG_PRIMITIVE => {
                        let ptype = PType::try_from(d[1] as i32)
                            .map_err(|_| vortex_err!("bad child ptype {}", d[1]))?;
                        DType::Primitive(ptype, nullability)
                    }
                    TAG_BOOL => DType::Bool(nullability),
                    TAG_UTF8 => DType::Utf8(nullability),
                    other => vortex_bail!("bad child descriptor tag {other}"),
                };
                let mode = if d[3] == MODE_REFERENCE {
                    ChildMode::Reference
                } else {
                    ChildMode::Values
                };
                let len_bytes: [u8; 8] = d[8..16]
                    .try_into()
                    .map_err(|_| vortex_err!("truncated child descriptor"))?;
                let len = usize::try_from(u64::from_le_bytes(len_bytes))?;
                Ok(ChildDescriptor { dtype, len, mode })
            })
            .collect()
    }
}

/// A [`GuestMem`] over a live kernel instance.
struct InstanceGuestMem<'a> {
    instance: &'a mut KernelInstance,
}

impl GuestMem for InstanceGuestMem<'_> {
    fn alloc(&mut self, len: u32) -> VortexResult<u32> {
        let ptr = self
            .instance
            .alloc
            .call(&mut self.instance.store, i32::try_from(len)?)
            .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} trapped: {e}"))?;
        u32::try_from(ptr).map_err(|_| vortex_err!("guest returned bad pointer"))
    }

    fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()> {
        self.instance
            .memory
            .write(&mut self.instance.store, off as usize, bytes)
            .map_err(|e| vortex_err!("failed to write guest memory: {e}"))
    }
}

fn map_trap(err: impl std::fmt::Display) -> VortexError {
    vortex_err!("wasm kernel trapped: {err}")
}
