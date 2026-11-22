# Short speakers

A design, not yet in the engine. Nemotron 3 has eight speaker numbers and
gives one to every voice it hears, in order of first speech. A podcast that
opens with a montage of clips, or carries ads read by voices heard for ten
seconds each, spends its numbers on them: of 100 episodes of two shows
(Pivot, 31; Prof G Markets, 69), 43 end with all eight in use, and in some
the eighth is taken in the first four minutes. After that the model has no
number for a guest who arrives late, and puts them on someone else's.

The design holds a voice with under a minute of speech out of the eight,
and costs, on those 100 episodes, 7.2% more diarizing on average and 40% at
most. It has two parts: a rewind while the recording plays, and a join
check at its end. Both were simulated outside the engine
(tools/shortspk/), on `pktito-c probs`; nothing here changes what
`pk_asr_run` or the diarizer's low level computes.

## The rewind

The diarizer's state between steps is about 0.6 MB, and a step is 27.2 s
(340 frames of 80 ms). The engine keeps the state at every step boundary.
At each boundary:

1. **Who is short.** A speaker with under 60 s of speech who has not been
   heard for 120 s starts a hold. Every other speaker under 60 s who has
   not been heard for 60 s is held with them, so one rewind serves a whole
   ad break.
2. **Only when numbers run low.** Nothing is held until seven numbers are
   in use. Most recordings never get there and pay nothing.
3. **Known again.** Before anyone is held, a copy of the state hears 1 s of
   silence and a 4 s clip of each speaker held earlier. A clip that lands
   in the column of a speaker about to be held (1 s or more, and half of
   what the copy heard) says they are the same voice. Their time adds up;
   at 60 s together the speaker keeps the number, the spans held earlier
   become theirs, and there is no rewind. A recurring ad voice is a speaker
   like any other once it has a minute.
4. **The rewind.** The held speakers' frames are cut from the audio (0.3 s
   each side, pauses under 2 s filled), the diarizer goes back to the state
   at the step before their first word, and hears the rest again without
   them. The cost is the seconds heard twice.
5. **A budget.** Seconds heard twice may not pass 0.25 of the recording so
   far plus 300 s. A rewind that would is skipped, and those speakers keep
   their numbers.

At the end, the speakers still held are played after the recording,
longest first, with 1 s of silence between, and take what numbers are
left. One that lands on no column (under 1 s, or under 20% of its
activity) is "other".

The constants are tools/shortspk/rewind.py's, chosen by hand on four
podcasts before the 100 were run; none was tuned on the 100.

## The join check

Nemotron 3 sometimes gives a speaker a second number late in a recording:
a host keeps one number for 40 minutes and takes another for a later
stretch. A plain run does this too, but less where all eight numbers are
taken, because there is no spare to hand out. Freeing numbers lets it show.
(The cause is read from when the second number appears, not confirmed in
the model.)

At the end of the recording:

1. A diarizer that has heard nothing hears three 4 s clips of every speaker
   with 60 s or more, one speaker after another.
2. Two speakers whose clips it puts in one column are heard again as a
   pair, on other clips.
3. If it says one speaker both times, the two numbers are joined.

## Measured

2026-11-22, M4 Pro and M5 Ultra, libpktito 6832680, the int8 Nemotron 3
model, `pktito-c probs`; 100 episodes, 31 to 80 minutes each.

**Cost**, as audio heard again over the recording's length:

| part                                    | median | mean | max   |
| --------------------------------------- | ------ | ---- | ----- |
| rewind from the first short speaker     | 24.1%  | 25.1 | 50.4% |
| rewind once seven numbers are in use    | 12.0%  | 14.2 | 38.2% |
| the same, over all 100 episodes         | 0.0%   | 7.2  | 38.0% |
| join check, over all 100                | 1.9%   | 1.9  | 3.5%  |

The first two rows are the 43 episodes that end with eight numbers in use.

**Who is right.** Where the plain run and the rewind run disagree about
two stretches of a minute or more being one speaker, three listeners heard
4 s clips of each: two speaker-embedding models (ReDimNet2-B3 and
WeSpeaker's ResNet34, cosine 0.40 or more for one voice) and a Nemotron 3
that had heard nothing else. They agree with each other on 1,012 of 1,027
pairs of stretches, and the two embedding models on all 1,027. Counting the
pairs where all three agree, over the 100 episodes with the rewind at seven
numbers:

|                                           | plain run | rewind run |
| ----------------------------------------- | --------- | ---------- |
| one voice on two numbers, in this run only | 16        | 14         |
| one voice on two numbers, in both runs     | 32        | 32         |
| two voices on one number                   | 14        | 0          |
| wrong, in all                              | 62        | 46         |

The rewind run is better in 13 episodes, the same in 84 and worse in 3.
Rewinding from the first short speaker makes about twice the new splits (23
to 11, on the 43) for twice the cost, which is why the design waits for
seven.

**The join check**, on the rewind run's output, against the embedding
model: of 36 pairs of long speakers that are one voice it joins 35, and of
385 that are two it joins 2, at cosines of 0.28 and 0.30 where two voices
are typically 0.10. On the plain run's output: 31 of 33, and 2 of 355.

**Against a person.** On 61 Prof G Markets episodes whose speakers a person
labelled by ear, before the join check: one host is on two numbers in 2
episodes of the plain run and none of the rewind run, the other in 7 and 8.

Not measured: the rewind and the join check scored together; any recording
that is not a two-host podcast; the cost in seconds on a small board.

## Into the engine

- The high level only. `pk_speakers` and the diarizer's steps stay as they
  are; the holding lives beside `pk_asr_transcribe`'s other processing, is
  documented as leaving NeMo's answer on purpose, and has a switch off.
- Memory the caller can bound: one 0.6 MB state per step is 1.3 MB a minute
  of audio. The budget bounds how far back a rewind can go, so states older
  than that can be dropped.
- The audio is needed again: the rewind replays up to the budget's worth,
  and the clips are 4 s each. A recording read from a pipe keeps that much.
- A held speaker's words are still transcribed. Until the end they have no
  number; what a caller shows for them meanwhile is the caller's.
- `tools/shortspk/rewind.py` is the reference for the rule, as
  tools/pktito_tools/reference.py is for the model: the C gives its
  speakers on the same recordings.
