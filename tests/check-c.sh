#!/bin/sh
# make check-c's commands (the Makefile says what each one holds), as a
# script so that a board without make runs the same ones: an Android device,
# over adb (tests/adb.sh). Run from anywhere with the programs built; MODEL
# is the model kcheck and the test.wav checks use. The first failure ends
# the run, as it ends make's.
cd "$(dirname "$0")/.." || exit 1
MODEL=${MODEL:-models/parakeet-tdt-0.6b-v2.pkf}
TINY=tests/data/tiny.pkf

run() {
    echo "$*"
    "$@" || exit 1
}
run ./tests/mathcheck
run ./tests/kcheck "$MODEL"
run ./tests/pkfcheck $TINY tests/pkfcheck.wav
run ./tests/wavcheck tests/wavcheck.tmp
run ./tests/pathcheck $TINY
run ./tests/capcheck
run ./tests/langcheck
run ./tests/chunkcheck
run ./tests/transcheck "$MODEL" tests/data/test.wav

./pktito-c -m $TINY --json tests/pkfcheck.wav > tests/pkfcheck.json \
  && ./pktito-c -m $TINY --srt tests/pkfcheck.wav > /dev/null \
  && ./pktito-c -m $TINY tests/pkfcheck.wav tests/pkfcheck.wav > /dev/null \
  && ./pktito-c -m $TINY --fast tests/pkfcheck.wav > /dev/null \
  && echo "pktito-c: the tiny model through the CLI, chunked, four ways" \
  || { echo "pktito-c: FAIL on the tiny model"; exit 1; }
d="tests/pathcheck-tést-ñ-я"; mkdir -p "$d" && cp $TINY "$d/tiny-é.pkf" && cp tests/pkfcheck.wav "$d/ñ.wav" \
  && ./pktito-c -m "$d/tiny-é.pkf" --json "$d/ñ.wav" | cmp -s - tests/pkfcheck.json; r=$?; rm -rf "$d"
[ $r = 0 ] && echo "pktito-c: a model and a WAV named past ASCII, from its command line" \
  || { echo "pktito-c: FAIL on a model and a WAV named past ASCII"; exit 1; }
rm -f tests/pkfcheck.wav tests/pkfcheck.json

./pktito-c probs -m "$MODEL" tests/data/test.wav | cmp -s - tests/data/test.probs.txt \
  && echo "test.wav: VAD probabilities are vadito's, bit for bit" \
  || { echo "test.wav: FAIL - VAD probabilities differ from tests/data/test.probs.txt"; exit 1; }
for v in v2 v3; do f=models/parakeet-tdt-0.6b-$v.pkf
  if [ ! -f $f ]; then echo "test.wav: no $f - skipped"; continue; fi
  ./pktito-c --json -m $f tests/data/test.wav | cmp -s - tests/data/test.$v.json \
    && echo "test.wav: $v transcript is ref's, every token and time" \
    || { echo "test.wav: FAIL - $v transcript differs from tests/data/test.$v.json"; exit 1; }
  ./pktito-c --fast --json -m $f tests/data/test.wav | cmp -s - tests/data/test.$v.fast.json \
    && echo "test.wav: $v transcript in the fast mode is ref's" \
    || { echo "test.wav: FAIL - $v --fast transcript differs from tests/data/test.$v.fast.json"; exit 1; }
done
./examples/transcribe "$MODEL" tests/data/test.wav | tr -d '\r' | cut -d' ' -f3- > tests/example.out
./pktito-c -m "$MODEL" tests/data/test.wav | cmp -s - tests/example.out \
  && echo "examples/transcribe: the CLI's transcript, through the library alone" \
  || { echo "examples/transcribe: FAIL - not the CLI's transcript"; exit 1; }
rm -f tests/example.out
