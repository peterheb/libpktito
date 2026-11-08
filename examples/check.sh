#!/bin/sh
# make check-ffi: each example whose language is here, built and run on
# test.wav, its transcript held to ./pktito-c's without the context and the
# retry, as check-c holds examples/transcribe's; one whose language is not
# here is skipped with a line. With ENCODER, an NPU's, the CLI (--npu) and
# every example, examples/transcribe too, run their encoder on it. Run from
# the top of the tree:
# examples/check.sh MODEL PYTHON [ENCODER]
model=$1 python=$2 encoder=$3 wav=tests/data/test.wav
# the examples write a chunk a line, the CLI a sentence: the words held in order
want=$(./pktito-c --no-context --no-retry -m "$model" ${encoder:+--npu "$encoder"} "$wav") || exit 1
want=$(printf '%s\n' "$want" | tr -s '\n' ' ')
status=0

# check NAME THROUGH COMMAND...: COMMAND MODEL WAV [ENCODER]'s text against the CLI's
check() {
    name=$1 through=$2
    shift 2
    if [ "$("$@" "$model" "$wav" ${encoder:+"$encoder"} | tr -d '\r' | cut -d' ' -f3- | tr -s '\n' ' ')" = "$want" ]; then
        echo "examples/$name: the CLI's transcript, through $through"
    else
        echo "examples/$name: FAIL - not the CLI's transcript"
        status=1
    fi
}
skip() { echo "examples/$1: $2 - skipped"; }
fail() { echo "examples/$1: FAIL - $2"; status=1; }

if [ -n "$encoder" ]; then
    check transcribe "the library alone" ./examples/transcribe
fi

if [ -n "$python" ]; then
    check python ctypes "$python" examples/python/transcribe.py
else
    skip python "no python3"
fi

if command -v go >/dev/null; then
    if (cd examples/go && go build); then
        check go cgo examples/go/transcribe
    else
        fail go "go build"
    fi
else
    skip go "no go"
fi

if command -v cargo >/dev/null; then
    # from its folder, where cargo finds its .cargo/config.toml
    if (cd examples/rust && cargo build --quiet --release); then
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
