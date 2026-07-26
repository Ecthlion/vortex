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
//! 3. the guest returns a decode plan — a small tree of operations over the node's children,
//!    which the host evaluates with its own lazy arrays.
//!
//! Kernels are untrusted file data. The runtime is `wasmtime` with its default Cranelift backend
//! (not Winch/Pulley, which are less battle-tested); each decode runs in a fresh [`Store`] whose
//! linear memory growth is capped via [`StoreLimits`]. CPU-time bounding (fuel / epoch
//! interruption) is a planned follow-up — see `docs/design/wasm-encodings.md`.

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
use crate::convert::CHILD_ENTRY_SIZE;
use crate::convert::GuestMem;
use crate::convert::write_child;
use crate::dtype as dtype_codec;
use crate::plan::Plan;

/// Maximum linear memory a kernel may grow to in a single decode, as a coarse DoS guard against
/// untrusted kernels. Generous enough for legitimate decodes (wasm32 memory tops out at 4 GiB); a
/// starting value, not a tuned one.
const MAX_GUEST_MEMORY_BYTES: usize = 1 << 30;

/// Cap on the number of children a kernel may declare.
const MAX_CHILDREN: usize = 4096;

/// Frame flag bit 0: the parent dtype is nullable.
///
/// The full dtype now rides in its own length-prefixed blob; this bit is kept because it is free
/// and it is what most kernels actually branch on.
const FLAG_NULLABLE: u32 = 1;

/// Encode the frame flags word for a node dtype.
fn frame_flags(dtype: &DType) -> u32 {
    if dtype.is_nullable() {
        FLAG_NULLABLE
    } else {
        0
    }
}

/// Fixed part of a child descriptor (see the guest SDK's `abi::child_descriptor`).
const DESCRIPTOR_HEADER: usize = 16;
const MODE_REFERENCE: u8 = 1;

/// Cap on the bytes a kernel may spend describing one child's dtype.
const MAX_CHILD_DTYPE_BYTES: usize = 4096;

/// How the kernel intends to use a serialized child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildMode {
    /// The kernel reads this child's element bytes: the host canonicalizes it and copies it into
    /// guest memory.
    Values,
    /// The kernel only names this child in its plan: the host resolves it lazily, in its own
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

/// Store state for a single decode: only the resource limiter (the ABI has no host callbacks that
/// carry state).
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
/// node has ([`children`](Self::children)), decode them, then run the decode step.
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
        let dtype_bytes = dtype_codec::encode(dtype)?;
        let mut frame = Vec::with_capacity(24 + dtype_bytes.len() + metadata.len());
        frame.extend_from_slice(&(len as u64).to_le_bytes());
        frame.extend_from_slice(&frame_flags(dtype).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(n_children)?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(dtype_bytes.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(metadata.len())?).to_le_bytes());
        frame.extend_from_slice(&dtype_bytes);
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
    ///
    /// Returns the kernel's plan, which the caller evaluates once it can resolve the node's
    /// serialized children.
    pub(crate) fn decode(
        &mut self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[ByteBuffer],
        children: &[Canonical],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Plan, Vec<u8>)> {
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
        let dtype_bytes = dtype_codec::encode(dtype)?;
        let mut frame = Vec::with_capacity(
            28 + dtype_bytes.len()
                + metadata.len()
                + buffers.len() * 8
                + child_entries.len() * CHILD_ENTRY_SIZE,
        );
        frame.extend_from_slice(&(len as u64).to_le_bytes());
        frame.extend_from_slice(&frame_flags(dtype).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(dtype_bytes.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(metadata.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(buffers.len())?).to_le_bytes());
        frame.extend_from_slice(&(u32::try_from(child_entries.len())?).to_le_bytes());
        frame.extend_from_slice(&dtype_bytes);
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

        // The result is a plan frame. Parsing checks its structure — node count, opcodes, and
        // that every operand points strictly backwards; the length and dtype checks happen when
        // it is evaluated, where the arrays exist.
        let mem = self.instance.memory.data(&self.instance.store);
        let plan = Plan::parse(mem, result_ptr as usize)?;
        // The plan's `Materialized` nodes point at buffers in guest memory, which dies with this
        // instance, so the caller gets a snapshot to build them from.
        Ok((plan, mem.to_vec()))
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

    /// Parse the `vx_children` result: `[u32 n]` then `n` variable-length descriptors, each a
    /// fixed header followed by a dtype expression resolved against `parent`.
    fn read_descriptors(&mut self, ptr: u32, parent: &DType) -> VortexResult<Vec<ChildDescriptor>> {
        let mem = self.memory.data(&self.store);
        let start = ptr as usize;
        let count = mem
            .get(start..start + 4)
            .ok_or_else(|| vortex_err!("failed to read the child descriptor count"))?;
        let n = usize::try_from(u32::from_le_bytes(
            count
                .try_into()
                .map_err(|_| vortex_err!("truncated child descriptor count"))?,
        ))?;
        vortex_ensure!(
            n <= MAX_CHILDREN,
            "wasm kernel declared too many children: {n}"
        );

        let mut offset = start + 4;
        (0..n)
            .map(|_| {
                let header = mem
                    .get(offset..offset + DESCRIPTOR_HEADER)
                    .ok_or_else(|| vortex_err!("truncated child descriptor"))?;
                let mode = if header[0] == MODE_REFERENCE {
                    ChildMode::Reference
                } else {
                    ChildMode::Values
                };
                let dtype_len = usize::try_from(u32::from_le_bytes(
                    header[4..8]
                        .try_into()
                        .map_err(|_| vortex_err!("truncated child descriptor"))?,
                ))?;
                vortex_ensure!(
                    dtype_len <= MAX_CHILD_DTYPE_BYTES,
                    "a kernel spent {dtype_len} bytes describing one child's dtype, over the \
                     {MAX_CHILD_DTYPE_BYTES} allowed"
                );
                let len = usize::try_from(u64::from_le_bytes(
                    header[8..16]
                        .try_into()
                        .map_err(|_| vortex_err!("truncated child descriptor"))?,
                ))?;
                offset += DESCRIPTOR_HEADER;
                let expr = mem
                    .get(offset..offset + dtype_len)
                    .ok_or_else(|| vortex_err!("truncated child dtype expression"))?;
                let (dtype, _) = dtype_codec::decode(expr, parent)?;
                offset += dtype_len;
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
