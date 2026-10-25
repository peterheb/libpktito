#!/bin/sh
# make adb: the loop an Android device gets, as make remote is a board's over
# ssh. Android has no make or compiler, so the tree is cross-built here with
# the NDK's clang for the device's ABI, in build/android-ARCH apart from this
# machine's own objects; the programs, the test data and the models go to
# ADIR on the device, and tests/check-c.sh and a bench run there. adb picks
# the device, over USB or Wi-Fi (ANDROID_SERIAL names one of several). The
# Makefile passes TESTS, MODEL, BENCH and BACKEND.
set -e
cd "$(dirname "$0")/.."
: "${NDK:?make adb NDK=/path/to/android-ndk}"
ADB=${ADB:-adb}
API=${API:-24}                          # Android 7.0, the oldest the build is tried on
ADIR=${ADIR:-/data/local/tmp/pktito}    # the one place adb's shell may write and run programs

abi=$($ADB shell getprop ro.product.cpu.abi | tr -d '\r')
case $abi in
  arm64-v8a) arch=aarch64 ;;
  x86_64) arch=x86_64 ;;
  *) echo "make adb: the device is $abi, not a family pktito builds"; exit 1 ;;
esac
bin=$(echo "$NDK"/toolchains/llvm/prebuilt/*/bin)
b=build/android-$arch

mkdir -p $b
rsync -a --delete --exclude /build --exclude .git --exclude .venv --exclude .venv-nemo \
  --exclude '*.o' --exclude '*.d' --exclude /pktito-c --exclude /pktito --exclude /models \
  --exclude '/tests/*check' ./ $b/
cross="CC=$bin/$arch-linux-android$API-clang AR=$bin/llvm-ar UNAME_M=$arch UNAME_S=Linux CPU_PART="
make -s -C $b $cross clean      # as remote does: nothing of this machine's build is taken as current
make -s -C $b -j8 $cross ${BACKEND:+BACKEND=$BACKEND} pktito-c $TESTS examples/transcribe

# what the device runs, laid out as the tree is, in one push
s=$b/push
rm -rf $s
mkdir -p $s/tests/data $s/examples
cp $b/pktito-c $s/
cp $b/examples/transcribe $s/examples/
for t in $TESTS; do cp $b/$t $s/tests/; done
cp tests/check-c.sh $s/tests/
cp tests/data/* $s/tests/data/
$ADB shell mkdir -p $ADIR/models
$ADB push --sync $s/. $ADIR/ | tail -1
for f in models/*.pkf $BENCH; do      # --sync: a model already there is not sent again
  case $f in tests/data/*) continue ;; esac
  [ -f "$f" ] || continue
  src=$f; [ -L "$f" ] && src=$(readlink -f "$f")    # the file, not a link adb would copy as one
  $ADB push --sync "$src" "$ADIR/$f" | tail -1
done
$ADB shell "cd $ADIR && MODEL=$MODEL sh tests/check-c.sh && ./pktito-c bench -m $MODEL $BENCH"
