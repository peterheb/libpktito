// libpktito.a, from the top of the tree (`make libpktito.a`) or PKTITO_LIB_DIR,
// and what it needs of its own: libm and pthreads. The archive is linked from
// a copy in OUT_DIR, where nothing else is: given -lpktito, macOS's linker
// takes libpktito.dylib over libpktito.a beside it, and `make shared` leaves
// one there.
fn main() {
    let dir = std::env::var("PKTITO_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../..", env!("CARGO_MANIFEST_DIR")));
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::copy(format!("{dir}/libpktito.a"), format!("{out}/libpktito.a"))
        .unwrap_or_else(|e| panic!("{dir}/libpktito.a: {e} (make libpktito.a at the top of the tree)"));
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=pktito");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed={dir}/libpktito.a");
    println!("cargo:rerun-if-env-changed=PKTITO_LIB_DIR");
}
