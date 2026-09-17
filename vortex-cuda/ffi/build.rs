// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let header = crate_dir.join("cinclude/vortex_cuda.h");
    // The CUDA API needs no macro expansion or dependency parsing, so generate on stable Rust
    // without recursively building the CUDA implementation.
    cbindgen::Builder::new()
        .with_src(crate_dir.join("src/lib.rs"))
        .with_config(cbindgen::Config::from_file(
            crate_dir.join("cbindgen.toml"),
        )?)
        .generate()?
        .write_to_file(&header);
    if !Command::new("clang-format")
        .args(["--style=file", "-i"])
        .arg(&header)
        .status()
        .is_ok_and(|status| status.success())
    {
        println!("cargo:warning=clang-format unavailable or failed; CUDA header left unformatted");
    }
    Ok(())
}
