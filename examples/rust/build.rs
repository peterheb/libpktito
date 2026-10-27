// libpktito.a, from the top of the tree (`make libpktito.a`) or PKTITO_LIB_DIR,
// and what it needs of its own: libm and pthreads.
fn main() {
    let dir = std::env::var("PKTITO_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../..", env!("CARGO_MANIFEST_DIR")));
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static=pktito");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed={dir}/libpktito.a");
    println!("cargo:rerun-if-env-changed=PKTITO_LIB_DIR");
}
