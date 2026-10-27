#!/bin/sh
# make check-ffi: each example whose language is here, built and run on
# test.wav, its transcript held to ./pktito-c's as check-c holds
# examples/transcribe's; one whose language is not here is skipped with a
# line. Run from the top of the tree: examples/check.sh MODEL PYTHON
model=$1 python=$2 wav=tests/data/test.wav
want=$(./pktito-c -m "$model" "$wav") || exit 1
status=0

# check NAME THROUGH COMMAND...: COMMAND MODEL WAV's text against the CLI's
check() {
    name=$1 through=$2
    shift 2
    if [ "$("$@" "$model" "$wav" | tr -d '\r' | cut -d' ' -f3-)" = "$want" ]; then
        echo "examples/$name: the CLI's transcript, through $through"
    else
        echo "examples/$name: FAIL - not the CLI's transcript"
        status=1
    fi
}
skip() { echo "examples/$1: $2 - skipped"; }
fail() { echo "examples/$1: FAIL - $2"; status=1; }

if [ -n "$python" ]; then
    check python ctypes "$python" examples/python/transcribe.py
else
    skip python "no python3"
fi

if command -v go >/dev/null; then
    if (cd examples/go && go build -o transcribe .); then
        check go cgo examples/go/transcribe
    else
        fail go "go build"
    fi
else
    skip go "no go"
fi

if command -v cargo >/dev/null; then
    if cargo build --quiet --release --manifest-path examples/rust/Cargo.toml; then
        check rust "extern \"C\"" examples/rust/target/release/transcribe
    else
        fail rust "cargo build"
    fi
else
    skip rust "no cargo"
fi

if ! command -v node >/dev/null; then
    skip node "no node"
elif [ ! -d examples/node/node_modules/koffi ]; then
    skip node "no koffi (npm install in examples/node)"
else
    check node koffi node examples/node/transcribe.mjs
fi

exit $status
