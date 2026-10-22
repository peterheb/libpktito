# Deterministic caption segmentation from word timestamps

**Status:** Implementation specification, version 1.1  
**Date:** 2026-10-22 (1.0: 2026-10-21)  
**Default profile:** `en-web-v1` — prerecorded, adult English, verbatim, sequential pop-on captions  
**Outputs:** A canonical cue list, diagnostics, and SRT/WebVTT exports

**Changes in 1.1** (pktito): `to` after a semi-modal is no phrase starter (5.2, 5.3), and a cue cut at a required pause in mid-sentence shows an ellipsis (6.1, 10, 11.2).

## 1. Recommendation

Implement caption segmentation as a **shortest-path problem over word boundaries**. Each edge represents one possible caption containing a contiguous range of words. Before scoring an edge, determine its final text, line break, start time, end time, and readability metrics. Dynamic programming then selects the best complete sequence.

Use audio pauses, punctuation, and simple grammatical rules to score boundaries. Enforce real layout and synchronization constraints. Preserve the transcript and report circumstances in which every word cannot fit within the requested readability limits.

This architecture is deterministic, explainable, and testable against an exhaustive solver on small inputs. It avoids both fixed-word-count chopping and the greedy approach of filling each caption until a limit is reached, which can leave a tiny, unreadable final caption.

The pipeline is:

1. Normalize and validate the transcript.
2. Compute safe boundaries, pause evidence, and linguistic features.
3. Enumerate structurally valid candidate captions.
4. Choose each candidate's line layout and final times.
5. Find the globally best partition using an additive score.
6. Validate, attach diagnostics, and serialize.

**Scope:** This specification covers offline segmentation and formatting. ASR, forced alignment, punctuation restoration, diarization, translation, and editorial text reduction are separate stages. No LLM is required in the segmentation kernel.

## 2. External guidance and product choices

There is no single universal set of subtitle limits. Use a named, versioned profile and document which requirements it implements.

| Primary guidance | Relevant recommendations |
|---|---|
| Netflix General Requirements [1] | At most two lines; generally use one when it fits. Duration from 5/6 second to seven seconds. Respect grammatical units when breaking lines. |
| Netflix English (USA) [2] | 42 characters per line; adult reading-speed ceiling of 20 characters per second (CPS), children's ceiling of 17 CPS. |
| Netflix Subtitle Timing Guidelines [3] | Synchronize with audio and picture; approximately half a second of post-speech display can help. Its delivery rules also require a two-frame intercue gap and account for shot changes. |
| DCMP Captioning Key [4, 5] | Phrase-aware line division; a 40-frame minimum in its stated 30 fps context and six-second maximum. Educational presentation-rate targets differ by audience: 130, 140, or 160 words per minute. |

The default below combines practical product choices. **It is not a claim of Netflix, DCMP, or broadcast-delivery compliance.** In particular, its zero minimum gap, one-second minimum duration, and short-utterance hold exception are product policies. SRT and WebVTT do not impose the same editorial limits as a delivery style guide.

All scoring weights and pause thresholds below are engineering defaults proposed by this specification. They should be calibrated against reviewed material, then versioned. The sources support the general priorities, not these particular weights.

### 2.1 Default configuration

```yaml
algorithm_version: caption-dp-v1
profile_id: en-web-v1
language: en
mode: verbatim

max_lines: 2
max_line_graphemes: 42
preferred_line_graphemes: 38

target_cps: 17
max_cps: 20
min_duration_ms: 1000
preferred_min_duration_ms: 1200
preferred_max_duration_ms: 6000
max_duration_ms: 7000

lead_in_ms: 0
preferred_tail_ms: 200
normal_max_tail_ms: 500
short_utterance_max_tail_ms: 1000
min_intercue_gap_ms: 0
close_gap_below_ms: 250

pause_weak_ms: 250
pause_strong_ms: 500
pause_required_cut_ms: 1000
pause_disagreement_gap_ms: 500
pause_disagreement_speech_ms: 250

speaker_change_requires_cut: true
shot_tail_cap: false
character_metric: unicode_extended_grapheme_cluster
unicode_version: PIN_TO_IMPLEMENTATION
detokenizer_version: english-space-v1
boundary_rules_version: english-boundaries-v1
scoring_version: caption-score-v1
```

Configuration validation must enforce integer values for counts, CPS rates, durations, and thresholds; positive character/rate/duration limits; `preferred_line_graphemes <= max_line_graphemes`; `target_cps <= max_cps`; ordered duration bands and pause thresholds; nonnegative tail/gap settings; and `short_utterance_max_tail_ms >= normal_max_tail_ms >= preferred_tail_ms`. Fractional CPS settings require an explicitly defined rational-number extension. V1 supports exactly one or two lines and fixed-onset timing; a nonzero lead-in requires a different timing implementation and version.

Formulas below use the named configuration values. Where explanatory text or worked formulas show 42, 38, 250, 500, 1000, or similar defaults, substitute the corresponding profile field when evaluating another profile. The integer scoring weights belong to `caption-score-v1` and change only with a scoring-version change.

The 17 CPS target gives the optimizer room below the 20 CPS ceiling. It is not the cited adult maximum. CPS and WPM measure different things: calculate both from actual text when useful; do not convert one using a universal average word length.

## 3. Input, normalization, and preservation

### 3.1 Kernel input

```typescript
type Word = {
  id: string;
  text: string;       // One orthographic word, with its attached punctuation.
  start_ms: number;  // Nonnegative integer on the media timeline.
  end_ms: number;    // Integer; strictly greater than start_ms.
  speaker_id?: string;
};

type BoundaryAnnotation = {
  before_word: number; // Internal boundary before words[k], 1 <= k < n.
  force_cut?: boolean;
  forbid_cut?: boolean;
  forbid_line_break?: boolean;
  soft_bind?: boolean; // E.g. inside a known personal name; penalty, not prohibition.
  sentence_end?: boolean; // When present, overrides sentence-end heuristic.
};

type SpeechPause = {
  start_ms: number;
  end_ms: number;
};

type Input = {
  words: Word[];
  media_duration_ms?: number;
  pauses?: SpeechPause[];
  pause_coverage?: { start_ms: number; end_ms: number }[];
  boundaries?: BoundaryAnnotation[];
  shot_changes_ms?: number[];
};
```

`pauses` means intervals classified as containing no speech. They may still contain music or other sounds. Their absence is not proof of acoustic silence. `pause_coverage` says where the detector actually ran; an empty pause list with full coverage means it found no pauses.

V1 expects English words joined with one ordinary space. A token containing several orthographic words must be adapted before entering the kernel, or explicitly treated as one indivisible display atom under a different detokenizer profile. Never distribute a multiword token's duration into fabricated word alignments inside this segmenter.

### 3.2 Adapter responsibilities

An ASR adapter may accept seconds instead of milliseconds, token-leading whitespace, or separate punctuation tokens. Its normalization must be explicit and versioned:

- Convert every timestamp with the **same monotone rounding function**, such as exact decimal round-half-up to milliseconds. Prefer integer milliseconds or decimal strings at the API boundary.
- Preserve original text and timestamps in the transcript store. The normalized display view may strip surrounding whitespace; record that transform.
- Attach punctuation-only ASR tokens to their appropriate word using the provider's detokenization rules. Preserve the mapping to source IDs. Do not interpret punctuation timestamps as new spoken events.
- Do not restore capitalization or punctuation in the kernel. If a separate model does so, store its resulting display transcript and annotation version.
- Preserve internal apostrophes, hyphens, punctuation, numbers, and spelling. Do not add an ellipsis merely because a caption ends.

Do not floor every start and ceil every end independently: that can create artificial overlaps between intervals that originally touched. Validate again after quantization; a positive source duration may collapse to zero milliseconds. Return a specific error instead of silently stretching it.

### 3.3 Validation

Reject input with non-finite or non-integer kernel timestamps, negative starts, `end_ms <= start_ms`, duplicate IDs, empty words, embedded whitespace/control characters, decreasing start times, or a word ending after a known media duration. Allow Unicode letters, combining marks, emoji, and punctuation; do not count their bytes as characters.

Require the transcript to be in its intended reading order with nondecreasing starts. **Do not silently sort it.** A word's end need not be monotone; overlapping alignments are handled below.

Validate auxiliary inputs before computing features:

- Require pause intervals and coverage to be supplied together, or neither. Both may be empty. All endpoints must be nonnegative integers, each interval must have positive duration, and intervals must remain inside a known media duration. Sort and merge overlapping/touching coverage intervals as well as pause intervals. Every pause must lie entirely within the union of detector coverage.
- Validate shot times as nonnegative integers within a known media duration, sort them, and remove duplicates. “First shot change” below always means earliest chronologically after this normalization.
- Accept at most one annotation per internal boundary index; reject duplicates and out-of-range/non-integer indices. An omitted boolean is false, except omitted `sentence_end`, which means use the heuristic. Validate that supplied flags are booleans. Boundaries `0` and `n` are immutable required endpoints and cannot be annotated.
- Report contradictory `force_cut`/`forbid_cut` flags as `BOUNDARY_CONFLICT` with `needs_editor`, as with a conflict discovered from timing. A `forbid_line_break` is independent of the cue-cut flags. Treat an empty speaker ID as invalid; omit the field for an unknown speaker.

For empty input, return `ok`, zero cues, an empty SRT, and `WEBVTT\n\n`.

When media duration is unknown, set the working upper bound to the largest word end and emit `MEDIA_DURATION_UNKNOWN`. This conservatively prevents inventing extra video after the last known word. Supplying the actual media duration allows the final caption to use real trailing time.

### 3.4 Preservation contract

The core operates on immutable normalized words and stores their IDs on every cue. Every word appears exactly once, in order. A line break replaces the separator between two words; a cue boundary changes presentation grouping; a pause marker (6.1) follows a cue's last word on screen. None of them changes words.

Check preservation through IDs and reconstructed canonical text, not by searching for repeated strings. Keep editorial changes, such as removing filler words or shortening speech, in a separately approved display-text revision with source mappings.

## 4. Boundaries and pauses

Use half-open word ranges. A candidate `[i, j)` contains words `i` through `j-1`. Boundary `k` lies before word `k`; boundaries `0` and `n` are the start and end of the transcript.

### 4.1 Safe sequential boundaries

For every prefix:

```text
prefix_end[0] = 0
prefix_end[k] = max(end_ms[t] for 0 <= t < k)
```

An internal boundary is temporally safe only if:

```text
prefix_end[k] <= words[k].start_ms
```

At a safe boundary:

```text
gap_ms[k] = words[k].start_ms - prefix_end[k]
```

This must use the **prefix maximum**, not just the preceding word's end. A long interval can overlap several later words. Those words form an indivisible temporal group for a sequential, nonoverlapping caption track.

An unsafe boundary may still be a line break: line layout does not create a new timed event. It cannot be a cue boundary in this profile.

At every unsafe internal boundary, initialize `pause_ms = 0`, `pause_bonus = 0`, and `pause_source = overlap`. Linguistic features still apply there, so `B(k)` and `cross(k)` remain defined for line layout and internal-cue scoring.

### 4.2 Pause evidence

Normalize detector intervals to the same timeline and timestamp quantizer as the words. Validate them, sort by `(start, end)`, and merge overlapping or touching pause intervals. Do not merge across a positive gap unless a separately versioned upstream detector explicitly does so.

For each safe internal boundary, let the available interval be:

```text
G = [prefix_end[k], words[k].start_ms]
```

Compute `pause_ms[k]` as follows:

1. If pause-detector coverage includes all of `G`, intersect each normalized pause with `G` and use the **longest continuous intersection**. Use zero if none exists.
2. Otherwise, use `gap_ms[k]` as a word-alignment proxy and mark the source `word_gap_proxy`. Do not combine partial detector coverage with guessed silence without an explicit alternative policy.

Store `pause_source`, raw gap, and chosen pause length. A word gap can reflect a real pause, missed speech, or alignment error. With full detector coverage, emit `TRANSCRIPT_AUDIO_DISAGREEMENT` when the raw gap is at least `pause_disagreement_gap_ms` and the longest continuous complement of the pause intervals inside that gap is at least `pause_disagreement_speech_ms`. That complement represents possible speech according to the detector, not proof that words were omitted. Segmentation cannot recover omitted speech.

Classify with inclusive thresholds:

| Pause length | Classification | Behavior |
|---:|---|---|
| `< 250 ms` | None | No pause preference |
| `250–499 ms` | Weak | Prefer a cut when other constraints agree |
| `500–999 ms` | Strong | Strong preference for a cut |
| `>= 1000 ms` | Required | Candidate cues must not cross the boundary |

Do not turn every hesitation into a separate caption. Weak and strong pauses compete with reading time, grammatical coherence, and the cost of another cue. The required-pause rule prevents a later phrase from appearing at the start of a cue that spans a long pause.

### 4.3 Required and forbidden cuts

A boundary requires a cut if it is a required pause, a supplied `force_cut`, or a change between two known unequal speaker IDs when the profile enables speaker cuts. An unknown speaker ID does not establish a speaker change.

A boundary forbids a cut if it is temporally unsafe or has a supplied `forbid_cut`.

Return `needs_editor` with `BOUNDARY_CONFLICT` if a boundary both requires and forbids a cut. Do not resolve overlapping speakers by pretending they spoke sequentially. A simultaneous-dialogue formatter is a separate feature; WebVTT permits overlapping cues, but this product profile does not [6].

Manual phrase locks should be rare. Use `soft_bind` for most automatically detected names and phrases so a genuine long hesitation can still split them.

### 4.4 Upstream audio analysis

The kernel accepts frozen pause intervals; it does not depend on one VAD package. A reproducible audio pipeline must record the decoded audio hash, channel-selection/downmix policy, resampling implementation, model/checkpoint hash, frame size, thresholds, hysteresis, interval-merging policy, padding, and time-offset mapping.

For segmentation tests, inject pause intervals directly. Test audio-to-pause detection separately using labeled clips. If an aligner stretches word intervals across a pause, this conservative algorithm will not cut through those intervals; repair alignment upstream instead of moving word times to satisfy caption layout.

## 5. Text measurements and linguistic features

### 5.1 Character count

For a range `[i, j)`, construct its **flat logical text** by joining normalized word texts with one U+0020 space. Define:

```text
C(i,j) = extended_grapheme_count(flat_text(i,j))
```

Count spaces and punctuation. Do not count export escaping, cue numbers, timecodes, formatting tags, or the inserted layout newline. For CPS, keep the original single separator even when it becomes a line break. This makes reading load independent of line layout.

Use a pinned implementation of Unicode extended grapheme segmentation [7]. In JavaScript, string `.length` counts UTF-16 code units; in other runtimes, byte or code-point counts also differ from grapheme counts. Native segmenters can change with runtime Unicode data, so pin the runtime/data or bundle a known implementation.

Measure each displayed line separately. Two 42-character lines may have a flat reading-load count of **85** because a joining space is replaced by the newline. Conversely, a flat count below 84 does not prove the words can be partitioned into two legal lines.

Grapheme count is a reproducible length proxy, not pixel width. A product-controlled renderer can add shaped-width checks using a pinned font, shaping engine, size, and viewport. Do not promise that 42 emoji or wide glyphs fit like 42 English letters. Other languages need their own detokenization, width, and reading-rate profiles.

### 5.2 Minimal deterministic English rules

The following deliberately simple rules make v1 implementable without NLP. They are heuristics, not a full grammatical analysis.

For boundary classification, remove trailing closing quotes/brackets `" ' ” ’ ) ] }` from the preceding token for inspection only. Preserve the original text. A terminal is `?`, `!`, or a final period, except a recognized abbreviation/initial or an ellipsis. Treat `…` and sequences of at least three final periods as a major pause mark, not automatic sentence completion.

The pinned case-insensitive abbreviation list is:

```text
mr. mrs. ms. dr. prof. sr. jr. st. vs. etc. e.g. i.e. a.m. p.m. u.s. u.k.
```

Also suppress sentence completion for a token matching a single ASCII capital followed by a period. A decimal point inside a number does not terminate a sentence; `3.14.` still has a separate terminal period. A provided `sentence_end` boolean overrides only this terminal decision.

Define `lex(token)` as Unicode case-folded token text after stripping leading/trailing characters from `.,;:!?…()[]{}"“”'‘’`. Internal apostrophes and hyphens remain. Use the same pinned Unicode data as the rest of the pipeline.

Use these fixed sets:

```text
determiners = a an the this that these those my your his her its our their
prepositions = about above across after against along among around at before
  behind below beneath beside between beyond by despite down during except
  for from in inside into near of off on onto out outside over past since
  through throughout to toward towards under underneath until up upon with
  within without
conjunctions = and but or nor for so yet because although though unless while
  if when than that as once until whether
pronouns = i you he she it we they who which
auxiliaries = am is are was were be been being have has had do does did can
  could may might must shall should will would not never

phrase_starters = and but or nor yet so because although though unless while
  if when after before until since as where which who to in on at for from
  with by of into over under between through during without

semi_modals = going have has had having got used ought want wants wanted need
  needs needed able supposed
```

Set `function_bond[k] = 1` when the preceding token's lexical form is in any of the first five sets and the token has no recognized trailing punctuation. Here, recognized trailing punctuation means a final character in `. , ? ! ; : — …` after the closing-quote/bracket stripping described above. Otherwise use zero. This discourages leaving `the`, `to`, or `would` at the end of a caption or line. Homographs and constructions such as a sentence-final auxiliary can fool this rule; frozen parser annotations may improve a later profile.

An optional `soft_bind` marks a boundary inside a known name, number/unit expression, or phrase. It adds a penalty but remains breakable. Do not infer that arbitrary adjacent capitalized words must be a name.

### 5.3 Boundary score

For each internal boundary, choose the first matching base value:

| Condition before/at boundary | `base[k]` |
|---|---:|
| Sentence terminal | 0 |
| Semicolon, colon, em dash, or ellipsis | 10 |
| Comma | 20 |
| Next token's lexical form is a phrase starter, other than `to` after a semi-modal | 25 |
| Otherwise | 60 |

Define `major_punctuation[k]` as the semicolon, colon, em-dash, or ellipsis category in this table. When a supplied sentence-end override is true, the terminal category takes precedence.

```text
pause_bonus[k] = 50 if pause_ms[k] >= 500
                 20 if pause_ms[k] >= 250
                  0 otherwise

B(k) = max(0, base[k] - pause_bonus[k])
       + 300 * function_bond[k]
       + 500 * soft_bind[k]
```

`B(k)` is a cost: lower is better. The grammatical penalties exceed the small visual-balance terms below. Charge zero for a required cue boundary and for the transcript end; there is no alternative to selecting those boundaries. For non-required cue boundaries, charge `B(k)`.

Also charge for crossing a useful internal boundary:

```text
cross(k) = (60 if sentence_terminal[k]
            else 15 if major_punctuation[k]
            else 0)
           + (80 if pause_ms[k] >= 500
              else 30 if pause_ms[k] >= 250
              else 0)
```

A required boundary cannot be crossed at any score. Ordinary sentence boundaries are soft: very short related utterances can share a cue when splitting would produce a flash.

## 6. Candidate line layout

For each candidate `[i,j)`:

1. If its flat text fits within `max_line_graphemes`, keep one line.
2. If it does not fit and `max_lines == 1`, reject the candidate. Otherwise enumerate every word boundary `k` with `i < k < j` that does not forbid a line break.
3. Construct lines from `[i,k)` and `[k,j)`, with no leading or trailing spaces.
4. Discard layouts where either line exceeds `max_line_graphemes`.
5. Select the minimum-cost layout, using the tie-break below.

For one line, with length `L`:

```text
layout_cost = 2 * max(0, L - preferred_line_graphemes)^2
```

For two lines:

```text
widows = number of lines that contain exactly one word AND fewer than 12 graphemes

shape_cost = 2 * (max(0, L1 - preferred_line_graphemes)^2
                  + max(0, L2 - preferred_line_graphemes)^2)
             + 8 * widows
             + ceildiv(abs(L1 - L2), 4)
             + 2 * indicator(L1 > L2)

layout_cost = B(k) + shape_cost
layout_key = (layout_cost, abs(L1 - L2), indicator(L1 > L2), k)
```

Choose the lexicographically smallest `layout_key`. Thus a slightly longer second line is only a mild preference. Never insert a line break inside a word in this profile.

### 6.1 Pause marker

A candidate `[i,j)` whose end `j` is a required pause (4.3), and whose last word has no recognized trailing punctuation (5.2), ends in the middle of a sentence because the speaker stopped there: trailed off, hesitated, or broke off. Show that with an ellipsis, U+2026, after its last word, the pause marker of the Netflix and DCMP style guides. A dash would claim an interruption the timing cannot tell from a hesitation. Count it as one grapheme in the last line, `L` or `L2` above, so that a layout that leaves no room for it is discarded like any other too-wide one, and in the cue's character count (8.2). Do not mark a cue that ends at a boundary that is not required, which is a sentence continuing in the next caption, or the track's last cue. Flag the cue; the words stay as they are, and the exporters draw the marker (10).

When no layout exists, reject that candidate. The optimizer may split the range into multiple captions. If even an indivisible word or temporal group cannot fit, return `needs_editor`; do not truncate a URL, change a name, or invent independently timed pieces of one word.

## 7. Candidate timing

**Important:** Compute final timing here, before scoring and optimization. Do not run a later pass that borrows time from selected neighboring cues without re-solving the optimization.

For candidate `[i,j)`, define:

```text
S = words[i].start_ms
A = max(words[t].end_ms for i <= t < j)  // Last covered audio end.
span = A - S
N = words[j].start_ms if j < n else infinity
T = media_duration_ms, or the conservative bound from section 3
```

If `i` is a safe start boundary, `A` can be obtained from `prefix_end[j]`. Retain the explicit maximum definition in tests.

### 7.1 Tail budget and minimum-duration exception

Most cues may remain up to 500 ms after their last word. Any very short cue may use additional available time, only as much as needed to approach the one-second minimum. There is no separate sentence-completion eligibility rule:

```text
minimum_extension = max(0, min_duration_ms - span)
H = max(normal_max_tail_ms,
        min(short_utterance_max_tail_ms, minimum_extension))
```

For a 200 ms `Yes.`, `H = 800 ms`. If the media and next utterance leave enough room, it can display for one second. This is an explicit product choice; a rigid 500 ms cap would make many brief replies fail the minimum even during a long pause.

The exception does not give every cue a one-second tail. Normal reading-speed extensions still use the 500 ms budget unless the minimum-duration calculation itself requires more.

### 7.2 Gap and end cap

For a non-final candidate, its endpoint must be safe, so `N >= A`:

```text
G = min(min_intercue_gap_ms, N - A)  // Preserve speech if requested gap cannot fit.
boundary_end = N - G
```

For a final candidate, use `G = 0` and `boundary_end = infinity`.

```text
cap = min(A + H, boundary_end, T, S + max_duration_ms)
```

If the optional shot-tail policy is enabled, further cap at the first supplied shot change `q` satisfying `A <= q`. A shot already crossed by the spoken content is not an automatic cut. This optional policy restricts trailing display only; it does not implement every delivery-specific shot-timing rule [3].

Reject the candidate if `cap < A`: that would require hiding text before its assigned speech finishes. A maximum-duration violation is never repaired by clipping covered words.

### 7.3 Desired end and closing tiny gaps

Use `ceildiv(x,y) = (x + y - 1) // y` for nonnegative integers and positive `y`.

```text
reading_target_ms = ceildiv(1000 * C, target_cps)

desired = max(A + preferred_tail_ms,
              S + min_duration_ms,
              S + reading_target_ms)

E = min(desired, cap)

if j < n and 0 < boundary_end - E < close_gap_below_ms:
    if boundary_end <= cap:
        E = boundary_end

D = E - S
```

This closes very short blank intervals when the allowed tail budget permits. It never consumes unlimited silence or crosses the next cue's onset. At exactly 250 ms, the default closing rule does not apply. A short blank interval may remain when a cap prevents extension; record it as a quality observation rather than exceeding the cap.

The default zero-gap profile allows adjacent cues to touch. For a frame-based delivery profile, derive its gap from the actual rational frame rate; two frames is not a universal 80 ms. Frame snapping must occur inside candidate construction, followed by validation, rather than after the DP.

### 7.4 Why the timing model composes

Every selected cue starts at its first word. For a cue ending at boundary `j`, the next selected cue necessarily starts at `words[j].start_ms`, regardless of how many words it will contain. Therefore the current edge's timing and score do not depend on the next edge's length.

Consequently, for consecutive selected cues:

```text
current.end_ms <= next.start_ms
```

This local timing policy makes an exact ordinary DP possible. More flexible joint lead-ins, time lending, or speed-dependent neighbor retiming require extra state or a separate joint optimization; do not keep the same recurrence and claim global optimality after arbitrary repairs.

## 8. Constraints, scoring, and optimization

### 8.1 Structural constraints versus readability goals

Every accepted candidate must preserve words, have a legal one/two-line layout, start/end at permitted boundaries, cross no required cut, cover its word intervals, stay inside the media, and last no longer than seven seconds. These are structural constraints for this implementation.

The following readability checks may be impossible even when those structural constraints hold:

```text
D >= min_duration_ms
1000 * C <= max_cps * D
next_start - E >= min_intercue_gap_ms  // Non-final cue only.
```

Perform the CPS comparison with integers. Do not approve a 20.04 CPS cue because its displayed UI value rounds to 20.0.

### 8.2 Additive violation severity

Use this exact integer score so that any fully compliant path beats every path with a readability violation:

```text
x = max(0, 1000 * C - max_cps * D)
f = max(0, min_duration_ms - D)
g = max(0, min_intercue_gap_ms - (N - E)) if j < n else 0

Vspeed = ceildiv(x * x, max_cps * max_cps * D)
Vshort = ceildiv(f * f, min_duration_ms)
Vgap   = ceildiv(g * g, max(1, min_intercue_gap_ms))

V = Vspeed + 4 * Vshort + 4 * Vgap
```

All durations are positive before this calculation. `Vspeed` is proportional to duration times squared relative excess reading rate. This avoids making “one bad cue” preferable merely because it produces fewer warning badges, and reduces the incentive to improve a score just by subdividing equally fast speech. `V = 0` exactly when the three readability checks pass.

Use sufficiently wide integer arithmetic, such as checked 64-bit arithmetic under validated profile/input bounds or arbitrary-precision integers. JavaScript implementations should use `BigInt` for the squared scores and accumulated path scores.

### 8.3 Aesthetic cost

For each structurally valid candidate:

```text
target_deficit = max(0, ceildiv(1000 * C, target_cps) - D)
short_preference = max(0, preferred_min_duration_ms - D)
long_preference  = max(0, D - preferred_max_duration_ms)

end_cost = 0 if j == n or boundary[j].requires_cut else B(j)

Q = 40                                      // Each new caption has a cost.
    + end_cost
    + sum(cross(k) for i < k < j)
    + layout_cost
    + 4 * ceildiv(target_deficit, 100)
    + 4 * ceildiv(short_preference, 100)
    + 2 * ceildiv(long_preference, 100)
```

The weights discourage fragmentation while rewarding strong boundaries and comfortable durations. A target rate is a ceiling preference, not a demand to accelerate slow, naturally paced speech.

### 8.4 Dynamic program

Create vertices `0..n`. A structurally valid candidate `[i,j)` is an edge `i -> j` with additive cost `(V, Q, 1)`.

```text
best[0] = (0, 0, 0)
best[1..n] = unreachable
parent[0..n] = none

for i in 0..n-1:
    if best[i] is unreachable:
        continue

    for j in candidate_endpoints_from(i):
        edge = make_candidate(i, j)
        if edge is structurally_invalid:
            continue

        proposal = best[i] + (edge.V, edge.Q, 1)

        if proposal is lexicographically smaller than best[j]:
            accept proposal and parent edge
        else if proposal equals best[j]:
            accept the path with the lexicographically smaller
            ordered vector of cue-end boundary indices

if best[n] is unreachable:
    return needs_editor with structural diagnostics

cues = reverse(backtrack(parent, n))
status = ok if best[n].V == 0 else needs_review
```

The final path comparison is:

```text
(sum_V, sum_Q, cue_count, ordered_cue_end_indices)
```

Each candidate's line layout already has a deterministic tie-break. For equal path lengths, lexicographic ordering of end-index vectors is preserved when a common suffix is appended, so retaining one best prefix is valid. Compare the actual vectors, or implement an equivalent proven persistent-path comparison. Hash iteration order and thread completion order must not decide ties.

This single DP is equivalent to searching for a fully readable path first and using an explicit severity-minimizing fallback only when none exists. “Globally best” means best under these fixed candidates, timing policy, and weights; it does not mean there is one objectively correct human segmentation.

Do not substitute a primary objective such as `(maximum_cue_violation, total_violation)` while keeping only one prefix per vertex. That comparison can lose optimal substructure when a later large violation makes earlier maxima tie. Use additive objectives as above, or a solver designed for a different objective.

### 8.5 Enumeration and performance

Enumerate all potential cue ends at safe, non-forbidden boundaries. For increasing `j`, stop when an already-internal required boundary would be crossed, the audio span exceeds the maximum duration, or text growth proves that no two-line layout can fit. With ordinary space-separated words, flat length above `2 * max_line_graphemes + 1` is such a proof. Always evaluate actual layout below that bound.

Do not reject a whole transcript simply because a one-word candidate is too short: it may become readable when combined with later words. Do not stop enumeration merely because one intermediate endpoint is unsafe or one candidate has no legal wrap; handle monotonic pruning separately from candidate rejection.

If `W` is the bounded number of words that can fit a candidate, the graph search is `O(nW)` after edge evaluation; enumerating all line splits gives roughly `O(nW²)` small layout evaluations. With 42-character lines, the English candidate window is naturally small. Account separately for text construction and Unicode segmentation. Safe prefix counts can optimize ordinary tokens, but do not assume byte counts or arbitrary Unicode substring counts are additive.

Precompute crossing-penalty prefix sums and prefix maxima. Cache layouts and candidate metrics. Avoid an arbitrary fixed “maximum 12 words” rule, which can exclude valid captions containing short words.

## 9. Output and failure behavior

```typescript
type Cue = {
  id: string;
  word_ids: string[];
  start_ms: number;
  end_ms: number;
  lines: string[];
  line_break_before_word?: number;
  char_count: number;
  duration_ms: number;
  cps_numerator: number;    // 1000 * char_count
  cps_denominator: number;  // duration_ms
  audio_start_ms: number;
  audio_end_ms: number;
  score: { violation: string; aesthetic: string }; // Decimal integers.
  diagnostics: Diagnostic[];
};

type Result = {
  status: "ok" | "needs_review" | "needs_editor" | "invalid_input";
  profile_id: string;
  algorithm_version: string;
  input_hash: string;
  cues: Cue[];
  diagnostics: Diagnostic[];
};
```

| Status | Meaning and product behavior |
|---|---|
| `ok` | A complete structurally valid path satisfies the configured readability checks. Other informational diagnostics may remain. Export normally. |
| `needs_review` | A complete faithful path exists, but minimum duration, reading ceiling, or requested gap cannot all be met under this model. Supply the cues and precise warnings. A general transcription product may export with a visible quality status; a strict delivery workflow should block certification. |
| `needs_editor` | No complete path meets structural constraints. Preserve the transcript and return the blocking regions. Do not silently return a partial file or claim a valid complete caption track. An explicitly selected overflow-draft feature can be separate. |
| `invalid_input` | The input contract is violated. Return stable error codes and affected IDs; require correction or an explicit adapter repair. |

Suggested diagnostics include `READING_SPEED_EXCEEDED`, `DURATION_TOO_SHORT`, `INTERCUE_GAP_TOO_SHORT`, `WORD_TOO_LONG`, `TEMPORAL_GROUP_TOO_WIDE`, `TEMPORAL_GROUP_TOO_LONG`, `BOUNDARY_CONFLICT`, `TIMESTAMP_COLLAPSED`, `INVALID_TIMESTAMP`, `TRANSCRIPT_AUDIO_DISAGREEMENT`, and `MEDIA_DURATION_UNKNOWN`.

For readability failures, include actual and configured values plus word IDs. For structural failure, report the earliest unreachable mandatory boundary/end, the first blocking reachability frontier or indivisible group, and the relevant candidate rejection reasons. An overlong word near the beginning should not surface only as an unexplained failure at EOF. A globally invalid input should not produce plausible-looking, silently incomplete subtitles.

### 9.1 What segmentation cannot fix

Rapid speech may contain more text than any synchronized, verbatim caption stream can display at 20 CPS. The available remedies are editorial reduction, different readability limits, extra display time when truly available, or a different presentation policy. Splitting the same text into more cues does not create more time.

Likewise, a plain word transcript lacks important non-speech sounds, music descriptions, and reliable speaker identification. This algorithm formats supplied dialogue; complete accessibility captioning needs those additional annotations. Do not invent them from missing recognized words.

## 10. Serialization

Export both formats from the same canonical integer-timestamp cue list. Use UTF-8, deterministic LF line endings, and exactly one blank line between blocks. Preserve chosen line breaks, and end a cue flagged with a pause marker (6.1) with U+2026 right after its last word. Do not rewrap or retime in the exporter.

### 10.1 SRT

```srt
1
00:00:01,000 --> 00:00:02,000
Hello world.

2
00:00:03,000 --> 00:00:04,000
Again.

```

Number cues consecutively starting at 1. Use `HH:MM:SS,mmm`, with at least two hour digits and no wrap at 24 hours. The Library of Congress documents this basic structure and notes that SRT is not well standardized [8]. Editorial line limits belong to the selected profile, not to this serialization grammar.

Because SRT players differ in handling HTML-like tags and entities, define and test an explicit payload policy for supported players. A practical `html_entities_v1` policy escapes literal `&`, `<`, and `>` and emits no styling tags, but universal literal rendering cannot be guaranteed across SRT players. Preserve original text internally and flag special-character interoperability where relevant. Test round trips against the declared policy, not an imaginary universal SRT parser.

### 10.2 WebVTT

```vtt
WEBVTT

1
00:00:01.000 --> 00:00:02.000
Hello world.

2
00:00:03.000 --> 00:00:04.000
Again.

```

Use the header, a blank line, unique optional cue IDs, and decimal-point millisecond separators. End times must exceed start times, and cues must be ordered by start. The format permits overlap; the default profile intentionally produces a sequential track [6].

For literal payloads, escape `&` first, then `<` and `>`. For example:

```text
logical text: A < B & C --> D
VTT payload:  A &lt; B &amp; C --&gt; D
```

Count the logical text before escaping. Do not send arbitrary transcript text through a markup path. Caption styling, voice tags, positions, and word-level karaoke timestamps can be separate exporter features.

## 11. Required automated tests

Separate **candidate tests**, **boundary/layout tests**, **whole-transcript tests**, and **optimizer tests**. A locally valid candidate need not be selected by the final optimizer.

### 11.1 Exact timing fixtures

These use the default profile, no detector intervals, and no shot caps.

| Fixture | Input words `(text, start, end)` and media end | Expected result |
|---|---|---|
| Basic phrase | `Hello,1000,1300`; `world.,1400,1800`; `T=4000` | One cue `1000–2000`, `Hello world.`, 12 characters, 12 CPS. |
| Required pause | `Hello.,1000,1600`; `Again.,3000,3600`; `T=5000` | Two cues `1000–2000` and `3000–4000`. The 1400 ms pause requires a cut. |
| Short standalone reply | `Yes.,1000,1200`; `T=4000` | One cue `1000–2000`; the minimum-duration tail exception permits 800 ms after speech. |
| Short final reply | `Yes.,1000,1200`; `T=1300` | One cue `1000–1300`; `needs_review`, 700 ms below the duration minimum. No media overrun. |
| Rapid speech | Three 10-character words at `0–150`, `150–300`, `300–450`; `T=450` | No fully readable path. Preserve all words and return `needs_review`. Merged candidate has 32 characters and about 71.11 CPS. |
| Two full lines, candidate only | Two 42-character words at `0–2500`, `2500–5000`; `T=6000` | Legal layout `42 / 42`, flat `C=85`, candidate `0–5200`. Do not reject by a total-84 rule. |

For the last case, the optimizer may choose another partition based on scores; assert the candidate metrics independently.

### 11.2 Boundary and layout fixtures

1. **Pause thresholds:** `249, 250, 499, 500, 999, 1000` ms classify as none, weak, weak, strong, strong, required. Soft preferences do not require a selected cut.
2. **Detector versus proxy:** words `Go,0,200` and `now.,1800,2200`; detector pauses `[200,900]` and `[1000,1700]`, with full coverage. The raw gap is 1600 ms, but the longest pause is 700 ms. With detector evidence the boundary is strong; without it the proxy makes it required. Do not sum disconnected intervals.
3. **Overlap chain:** `A,0,2000`; `B,500,700`; `C,1000,1300`; `D,2100,2400`. Cuts after A and B are unsafe; the cut after C is safe. Candidate A–C ends its audio at 2000, not 1300.
4. **Touching endpoints:** `A,0,300`; `B,300,600` permits a boundary. Extending A to 301 forbids it. A requested gap never justifies trimming A's speech.
5. **Impossible long token:** one 43-grapheme word cannot fit this profile, even though two lines are allowed. Return a structural error with the word intact.
6. **Total length is insufficient:** word lengths `30,30,20` give flat length 82, but the two possible wraps are `30 / 51` and `61 / 20`. Reject that full candidate; several separate cues may still be feasible.
7. **Unicode:** `cafe\u0301` counts as four graphemes; 42 repetitions of `e\u0301` fit a single line; 43 do not. A family emoji joined with ZWJs is one grapheme under the pinned rules. Add visual-width tests separately.
8. **Punctuation:** `Dr.` and a capital initial do not mark automatic sentence completion; a decimal's interior dot does not; `Really?` does. A supplied sentence-end annotation overrides the heuristic.
9. **No punctuation:** an unpunctuated transcript still segments from timing, width, and pauses. It must not acquire invented capitals or periods, and an ellipsis only at a required pause in mid-sentence (6.1): `So,54710,55030`, `I,55030,55350`, `when,56550,56800` puts `So I…` in a cue of its own; with `I,` instead of `I`, no ellipsis.
10. **Speaker overlap conflict:** A says `Wait.,0,1200`; B says `No.,900,1500`. The default speaker cut conflicts with the unsafe temporal boundary; return `needs_editor`.
11. **Gap closing:** evaluate otherwise identical candidates leaving 249 ms and 250 ms to `boundary_end`. Close only the former, and only when the full extension fits the cap. Recalculate all metrics afterward.
12. **Input errors:** parameterize negative/non-finite/reversed/zero timestamps, duplicate IDs, decreasing starts, words beyond media end, empty text, and collapsed durations after rounding. Expect stable error codes.

### 11.3 Global-versus-greedy regression

Create nine eight-character words, each occupying the next 500 ms, with no gaps and media end 4500 ms. No word has punctuation or a grammatical bond.

A two-line cue can hold eight such words as four per line; five on one line would be 44 characters. Greedy maximum packing chooses eight words followed by one, leaving a final 500 ms flash.

A compliant path exists: five words at `0–2500` and four at `2500–4500`, with reading rates 17.6 and 17.5 CPS. Other compliant splits may exist. Assert that the DP finds `sum_V = 0` and does not choose the eight-plus-one partition. The final exact partition follows the documented score.

### 11.4 Exhaustive oracle

For small transcripts, for example up to 12 words, enumerate every possible partition, discard structurally invalid candidates, and calculate the complete comparison tuple. Compare the DP output to the exhaustive minimum, including boundary-vector ties.

Use a separately implemented enumerator. Candidate calculations should have their own exact fixtures so both solvers cannot conceal the same timing or counting bug. Include feasible and infeasible cases, required/forbidden cuts, overlaps, and reversed candidate enumeration order.

Also test a synthetic equal-cost graph with paths `0→1→4` and `0→2→4`. With equal cue count, the specified boundary tie-break selects `[1,4]` regardless of insertion order.

### 11.5 Property tests

- Concatenating cue word IDs reproduces the exact input ID sequence.
- Reconstructing canonical text from cue words reproduces the normalized transcript.
- Every cue covers every assigned normalized word interval.
- Cues have positive duration, remain within the media, and do not overlap.
- Every required boundary is selected; no forbidden boundary is selected.
- Every output layout satisfies the structural line constraints.
- `ok` implies every configured readability check passes.
- `needs_review` reports every violated readability check with exact values.
- Each edge's times and score are unchanged when only the following candidate's length changes.
- Identical normalized inputs, annotations, and versions produce byte-identical SRT, VTT, and canonical diagnostics.
- Export/parse round trips preserve times and text under the declared payload policy, including `00:59:59.999 → 01:00:00.001`, literal markup characters, and files longer than 24 hours.

Exclude creation timestamps, unordered maps, and run-specific IDs from deterministic output, or store them outside the hashed/exported artifact.

### 11.6 Validation performed for this specification

A scratch ASCII reference prototype was checked against independent exhaustive partition enumeration for 2,000 seeded transcripts of 1–10 words, including gaps, overlaps, punctuation, boundary annotations, and nonzero requested intercue gaps. Every DP objective and boundary-vector tie matched the exhaustive optimum. Reversed/shuffled candidate enumeration preserved results, and all 24 insertion orders of the synthetic tie graph selected the specified path.

Separate exact fixtures checked the principal candidate-timing and layout cases above. For the nine-word greedy regression, the complete score selects `4 + 5` words with `(sum_V, sum_Q, cue_count) = (0, 211, 2)`; it wins the tie with `5 + 4` through the earlier first boundary.

This verifies the reference formulation, not a production implementation or human readability. The prototype uses ASCII counts; a shipping implementation must still run the Unicode, exporter/player, and human-review acceptance tests specified here.

## 12. Product integration and acceptance criteria

Store the immutable word alignment separately from editable caption boundaries and display text. Persist the profile, algorithm version, annotation versions, detector provenance, and input hash with each generated track. User locks become explicit boundary constraints on subsequent runs.

Display per-cue warnings with a waveform and source-word selection. “Too fast: 27.4 CPS, target ceiling 20” is actionable; a generic “AI quality issue” is not. Keep rejected alternatives and score components available in a debug trace for development.

Evaluate a reviewed corpus covering continuous narration, fast technical speech, interviews, short replies, hesitant speech, long pauses, numbers/names, imperfect punctuation, overlapping alignments, and clip edges. Record reading-rate distribution, short-cue count, line violations, phrase-break judgments, pause alignment, and edit distance from reviewed boundaries. There can be several good segmentations; boundary agreement alone is not a complete quality metric.

Release v1 when:

1. Exact fixtures, property tests, serialization tests, and the exhaustive optimizer oracle pass.
2. Input and structural failures never silently discard words or fabricate timing.
3. All readability exceptions are exposed in a machine-readable result.
4. Reviewers find the resulting captions comfortable across the evaluation corpus at normal playback speed.
5. Export rendering has been checked in the product's actual supported players and target viewport sizes.

A later version can add language-specific parsers, rendered-width constraints, dual-speaker layout, scene-aware segmentation, or joint timing optimization. Keep each enhancement behind a new profile/algorithm version and preserve the same conservation and diagnostic guarantees.

## References

Primary sources consulted 2026-10-21. Source recommendations are summarized above; the proposed scoring and algorithm are original engineering choices in this specification.

1. Netflix, **Timed Text Style Guide: General Requirements**. Duration and line treatment. <https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements>
2. Netflix, **English (USA) Timed Text Style Guide**. Character limits and reading-speed ceilings. <https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide>
3. Netflix, **Timed Text Style Guide: Subtitle Timing Guidelines**. Audio alignment, trailing time, gaps, and shot changes. <https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines>
4. DCMP, **Captioning Key — Text**. Line division and caption duration. <https://dcmp.org/learn/597-captioning-key---text>
5. DCMP, **Captioning Key — Presentation Rate**. Audience-specific rates and editorial considerations. <https://dcmp.org/learn/601-captioning-key---presentation-rate>
6. W3C, **WebVTT: The Web Video Text Tracks Format**, Candidate Recommendation Draft. File syntax and cue constraints. <https://www.w3.org/TR/webvtt1/>
7. Unicode, **UAX #29: Unicode Text Segmentation**. Extended grapheme clusters; pin a specific implementation/data version. <https://www.unicode.org/reports/tr29/>
8. Library of Congress, **SubRip Subtitle format (SRT)**. Serialization structure and interoperability context. <https://www.loc.gov/preservation/digital/formats/fdd/fdd000569.shtml>
