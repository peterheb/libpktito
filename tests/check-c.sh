#!/bin/sh
# make check-c's commands (the Makefile says what each one holds), as a
# script so that a board without make runs the same ones: an Android device,
# over adb (tests/adb.sh). Run from anywhere with the programs built; MODEL
# is the model kcheck and the test.wav checks use. The first failure ends
# the run, as it ends make's. tests/check-c.ps1 has the same commands for
# Windows without sh, and a check added here goes there too.
cd "$(dirname "$0")/.." || exit 1
MODEL=${MODEL:-models/parakeet-tdt-0.6b-v2.pkf}
TINY=tests/data/tiny.pkf
TINY_DIAR=tests/data/tiny-diar.pkf

run() {
    echo "$*"
    "$@" || exit 1
}
run ./tests/mathcheck
run ./tests/kcheck "$MODEL"
run ./tests/q8check
run ./tests/pkfcheck $TINY tests/pkfcheck.wav
run ./tests/pkfcheck $TINY_DIAR
run ./tests/wavcheck tests/wavcheck.tmp
run ./tests/pathcheck $TINY
run ./tests/capcheck
run ./tests/langcheck
run ./tests/editioncheck tests/editions.tsv
run ./tests/chunkcheck
run ./tests/transcheck "$MODEL" tests/data/test.wav
run ./tests/sentcheck "$MODEL" tests/data/test.wav

./pktito-c -m $TINY --json tests/pkfcheck.wav > tests/pkfcheck.json \
  && ./pktito-c -m $TINY --srt tests/pkfcheck.wav > /dev/null \
  && ./pktito-c -m $TINY tests/pkfcheck.wav tests/pkfcheck.wav > /dev/null \
  && ./pktito-c -m $TINY --fast tests/pkfcheck.wav > /dev/null \
  && echo "pktito-c: the tiny model through the CLI, chunked, four ways" \
  || { echo "pktito-c: FAIL on the tiny model"; exit 1; }
d="tests/pathcheck-tést-ñ-я"; mkdir -p "$d" && cp $TINY "$d/tiny-é.pkf" && cp tests/pkfcheck.wav "$d/ñ.wav" \
  && ./pktito-c -m "$d/tiny-é.pkf" --json "$d/ñ.wav" | cmp -s - tests/pkfcheck.json \
  && PKTITO_MODEL="$d/tiny-é.pkf" ./pktito-c --json "$d/ñ.wav" | cmp -s - tests/pkfcheck.json; r=$?; rm -rf "$d"
[ $r = 0 ] && echo "pktito-c: a model and a WAV named past ASCII, from its command line and \$PKTITO_MODEL" \
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
# --stream resamples as it reads: at 8 kHz (a telephone's) and 48 kHz (most
# microphones'), the lines of the same samples resampled first
s=tests/streamcheck
for hz in 8000 48000; do
  ./tests/streamcheck tests/data/test.wav $hz $s-in.wav $s-16k.wav \
    && ./pktito-c --stream --json -m "$MODEL" $s-in.wav > $s-in.json && [ -s $s-in.json ] \
    && ./pktito-c --stream --json -m "$MODEL" $s-16k.wav | cmp -s - $s-in.json \
    && echo "pktito-c --stream: test.wav at $hz Hz, resampled as it streams, as resampled first" \
    || { echo "pktito-c --stream: FAIL at $hz Hz - not the lines of the same samples resampled first"; exit 1; }
done
rm -f $s-in.wav $s-16k.wav $s-in.json
# the example writes a chunk a line, the CLI a sentence: the words held in order
./examples/transcribe "$MODEL" tests/data/test.wav | tr -d '\r' | cut -d' ' -f3- | tr -s '\n' ' ' > tests/example.out
./pktito-c --no-context --no-retry -m "$MODEL" tests/data/test.wav | tr -s '\n' ' ' | cmp -s - tests/example.out \
  && echo "examples/transcribe: the CLI's transcript without the context and the retry, through the library alone" \
  || { echo "examples/transcribe: FAIL - not the CLI's transcript"; exit 1; }
rm -f tests/example.out
