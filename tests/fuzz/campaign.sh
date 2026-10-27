#!/bin/sh
# Every fuzz target at once, for good, in cycles: build, fuzz each target for
# CYCLE seconds on its share of WORKERS, sort what crashed by where it
# crashed, minimize each corpus, write STATUS, and go again.
#
# A crash is replayed and named by its sanitizer's kind and the first three
# frames inside the tree; the first input of each kind is kept under
# findings/ and the rest are counted and deleted. A run with -ignore_crashes
# that hit one bug used to leave an artifact for every input that reached
# it (167k for one bug in k_quantize), so a target that writes more
# than FLOOD artifacts in a cycle is stopped for the rest of it.
#
#   tests/fuzz/campaign.sh                 tests/fuzz/README.md has a unit
#
# Environment (defaults suit 16 threads and 32 GB):
#   FUZZ_HOME  where corpora, findings and logs live ($HOME/pktito-fuzz)
#   CYCLE      seconds a cycle fuzzes (3600)
#   WORKERS    target:workers ... (the slow inference targets get the most)
#   UPDATE     1: git pull --ff-only before each cycle's build (0)
#   FLOOD      artifacts a target may write in a cycle before it is stopped (1000)
#   RSS        megabytes a worker may use (1536)
#   NOTIFY     a command, run with a new finding's summary as its last argument
set -u

REPO=$(cd "$(dirname "$0")/../.." && pwd)
FUZZ_HOME=${FUZZ_HOME:-$HOME/pktito-fuzz}
CYCLE=${CYCLE:-3600}
WORKERS=${WORKERS:-"model_semantic:5 audio_pipeline:4 state:2 pkf_bytes:1 wav_bytes:1 resample:1"}
UPDATE=${UPDATE:-0}
FLOOD=${FLOOD:-1000}
RSS=${RSS:-1536}
NOTIFY=${NOTIFY:-}

export PK_FUZZ_MODEL="$REPO/tests/data/tiny.pkf"
export ASAN_OPTIONS="detect_leaks=1${ASAN_OPTIONS:+:$ASAN_OPTIONS}"
export UBSAN_OPTIONS="print_stacktrace=1${UBSAN_OPTIONS:+:$UBSAN_OPTIONS}"

BIN=$REPO/build/fuzz
SEEDS=$FUZZ_HOME/seeds
CORPUS=$FUZZ_HOME/corpus
ART=$FUZZ_HOME/artifacts
FIND=$FUZZ_HOME/findings
LOGS=$FUZZ_HOME/logs
RUN=$FUZZ_HOME/run
mkdir -p "$SEEDS" "$CORPUS" "$ART" "$FIND" "$LOGS" "$RUN"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" >> "$LOGS/campaign.log"; }

targets() { for tw in $WORKERS; do echo "${tw%%:*}"; done; }

stop() {
    for f in "$RUN"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done
    wait
    log "stopped"
    exit 0
}
trap stop INT TERM

# make fuzz writes each target's seeds under FUZZ_CORPUS; they are read
# beside the live corpus every cycle and never written to
build() {
    if [ "$UPDATE" = 1 ] && ! git -C "$REPO" pull -q --ff-only >> "$LOGS/build.log" 2>&1; then
        log "git pull failed; staying on $(git -C "$REPO" rev-parse --short HEAD)"
    fi
    if make -s -C "$REPO" fuzz FUZZ_CORPUS="$SEEDS" >> "$LOGS/build.log" 2>&1; then
        return 0
    fi
    log "build failed (logs/build.log)"
    for t in $(targets); do [ -x "$BIN/$t" ] || return 1; done
    log "fuzzing the binaries from the last build"
}

fuzz() {
    for tw in $WORKERS; do
        t=${tw%%:*} w=${tw##*:}
        mkdir -p "$CORPUS/$t" "$ART/$t"
        max=$(make -s -C "$REPO" fuzz-max FUZZ="$t")
        (cd "$LOGS" && exec "$BIN/$t" "$CORPUS/$t" "$SEEDS/$t" -fork="$w" -ignore_crashes=1 \
            -max_total_time="$CYCLE" -max_len="$max" -timeout=10 -rss_limit_mb="$RSS" \
            -artifact_prefix="$ART/$t/" -print_final_stats=1 -verbosity=0) > "$LOGS/$t.log" 2>&1 &
        echo $! > "$RUN/$t.pid"
    done
    log "fuzzing $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) for ${CYCLE}s: $WORKERS"
    while :; do
        alive=0
        for t in $(targets); do
            pid=$(cat "$RUN/$t.pid" 2>/dev/null) || continue
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$RUN/$t.pid"
                continue
            fi
            alive=1
            n=$(ls "$ART/$t" | wc -l)
            if [ "$n" -gt "$FLOOD" ]; then
                log "$t: $n artifacts this cycle; stopped until the next"
                kill "$pid" 2>/dev/null
            fi
        done
        [ "$alive" = 1 ] || break
        sleep 30
    done
    wait
}

# The sanitizer's kind and the first three frames in the tree of the first
# stack (not ASan's allocation stack after it), columns dropped: one bug
# reached from many inputs gives the same text.
signature() {
    out=$("$BIN/$1" -runs=1 -timeout=10 -rss_limit_mb="$RSS" "$2" 2>&1)
    echo "$out" | sed -n -E -e 's/^SUMMARY: AddressSanitizer: .* leaked in .*/LeakSanitizer: leak/p' \
        -e 's/^SUMMARY: ([A-Za-z]+: [a-z-]+( signal)?).*/\1/p' | head -n 1
    echo "$out" | awk '/^ *#[0-9]+ 0x/ { seen = 1; print; next } seen { exit }' \
        | sed -n -E 's/^ *#[0-9]+ 0x[0-9a-f]+ in ([^ ]+) .*[/ ]((core|backend|tests)\/[^ :]+:[0-9]+).*/\1 \2/p' \
        | head -n 3
}

triage() {
    for t in $(targets); do
        for a in "$ART/$t"/*; do
            [ -f "$a" ] || continue
            sig=$(signature "$t" "$a")
            [ -n "$sig" ] || sig="does not reproduce: $(basename "$a" | cut -d- -f1)"
            id=$t-$(echo "$sig" | cksum | cut -d' ' -f1)
            if [ ! -d "$FIND/$id" ]; then
                mkdir -p "$FIND/$id"
                cp "$a" "$FIND/$id/input"
                { echo "$sig"; echo "first seen $(date '+%Y-%m-%dT%H:%M:%S') at $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"; } > "$FIND/$id/signature"
                echo 0 > "$FIND/$id/count"
                log "NEW $id: $(echo "$sig" | head -n 2 | tr '\n' ' ')"
                [ -z "$NOTIFY" ] || $NOTIFY "pktito fuzz: $t: $(echo "$sig" | head -n 2 | tr '\n' ' ')" || log "NOTIFY failed"
            fi
            echo $(( $(cat "$FIND/$id/count") + 1 )) > "$FIND/$id/count"
            rm -f "$a"
        done
    done
}

# -merge=1 keeps the inputs that add coverage, so the corpus stays the
# smallest set that reaches what the campaign has reached
minimize() {
    for t in $(targets); do
        (
            max=$(make -s -C "$REPO" fuzz-max FUZZ="$t")
            rm -rf "$CORPUS/$t.new" && mkdir "$CORPUS/$t.new"
            if "$BIN/$t" -merge=1 -max_len="$max" -timeout=10 -rss_limit_mb="$RSS" \
                   "$CORPUS/$t.new" "$CORPUS/$t" "$SEEDS/$t" > "$LOGS/$t.merge.log" 2>&1; then
                rm -rf "$CORPUS/$t.old" && mv "$CORPUS/$t" "$CORPUS/$t.old" \
                    && mv "$CORPUS/$t.new" "$CORPUS/$t" && rm -rf "$CORPUS/$t.old"
            else
                log "$t: merge failed (logs/$t.merge.log); corpus kept"
                rm -rf "$CORPUS/$t.new"
            fi
        ) &
    done
    wait
}

status() {
    {
        echo "$(date '+%Y-%m-%dT%H:%M:%S') at $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
        for t in $(targets); do
            last=$(grep -E '^#[0-9]+' "$LOGS/$t.log" 2>/dev/null | tail -n 1 | sed -E 's/^(#[0-9]+).*(cov: [0-9]+ ft: [0-9]+).*/\1 \2/')
            printf '%-15s corpus %6s  %s\n' "$t" "$(ls "$CORPUS/$t" | wc -l | tr -d ' ')" "$last"
        done
        echo "findings:"
        for d in "$FIND"/*; do
            [ -d "$d" ] && printf '  %-28s %6s  %s\n' "$(basename "$d")" "$(cat "$d/count")" "$(head -n 1 "$d/signature")"
        done
    } > "$FUZZ_HOME/STATUS"
}

log "start: FUZZ_HOME=$FUZZ_HOME CYCLE=$CYCLE WORKERS=$WORKERS UPDATE=$UPDATE"
while :; do
    if ! build; then
        sleep 600
        continue
    fi
    fuzz
    triage
    minimize
    status
done
