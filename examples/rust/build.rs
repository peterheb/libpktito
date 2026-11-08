// libpktito.a, from the top of the tree (`make libpktito.a`) or PKTITO_LIB_DIR,
// and what it needs of its own: libm and pthreads, and with an NPU what
// build.opts, which make writes beside it, says it was built with: Core ML
// and Foundation for ANE=1, libdl for QNN=1 off Windows. The archive is linked from
// a copy in OUT_DIR, where nothing else is: given -lpktito, macOS's linker
// takes libpktito.dylib over libpktito.a beside it, and `make shared` leaves
// one there.
fn main() {
    let dir = std::env::var("PKTITO_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../..", env!("CARGO_MANIFEST_DIR")));
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::copy(format!("{dir}/libpktito.a"), format!("{out}/libpktito.a")).unwrap_or_else(|e| {
        panic!("{dir}/libpktito.a: {e} (make libpktito.a at the top of the tree)")
    });
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=pktito");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    let opts = std::fs::read_to_string(format!("{dir}/build.opts")).unwrap_or_default();
    let with = |o| opts.split_whitespace().any(|x| x == o);
    if with("ANE=1") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    if with("QNN=1") && std::env::var("CARGO_CFG_WINDOWS").is_err() {
        println!("cargo:rustc-link-lib=dl");
    }
    println!("cargo:rerun-if-changed={dir}/libpktito.a");
    println!("cargo:rerun-if-changed={dir}/build.opts");
    println!("cargo:rerun-if-env-changed=PKTITO_LIB_DIR");
}
