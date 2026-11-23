# Short speakers

Nemotron 3 has eight speaker numbers and gives one to every voice it
hears, in order of first speech. A podcast that opens with a montage of
clips, or carries ads read by voices heard for ten seconds each, spends its
numbers on them: of 100 episodes of two shows (Pivot, 31; Prof G Markets,
69), 43 end with all eight in use. After that the model has no number for a
guest who arrives late, and puts them on someone else's.

`pk_diar_run_held` (core/hold.c) holds a voice with under a minute of
speech out of the eight, and joins two numbers that are one voice. It has
two parts: a rewind while the recording plays, and a join check at its end.
On those 100 episodes it hears 9.9% more audio than the recording on
average and 44% at most. pktito-c's speakers go by it, and `--no-hold`
gives the model's own rows; nothing here changes what `pk_diar_run`
computes. tools/shortspk/rewind.py is the reference for the rule, as
tools/pktito_tools/reference.py is for the model: on the 100 episodes the
C's rows are its rows bit for bit, and the stretches held the same.

## The rewind

The diarizer's state between steps is about 0.6 MB (`pk_diar_save`), and a
step is 27.2 s (340 frames of 80 ms). The engine keeps the state at every
step boundary. At each boundary:

1. **Who is short.** A speaker with under 60 s of speech who has not been
   heard for 120 s starts a hold. Every other speaker under 60 s who has
   not been heard for 60 s is held with them, so one rewind serves a whole
   ad break.
2. **Only when numbers run low.** Nothing is held until seven numbers are
   in use. Most recordings never get there and pay nothing.
3. **Known again.** Before anyone is held, the diarizer as it was a step
   ago hears the step again, then 1 s of silence and a 4 s clip of each
   speaker held earlier (a step ago, because the chunk that ends at the
   boundary has already heard 3.2 s past it). A clip that lands
   in the column of a speaker about to be held (1 s or more, and half of
   what the copy heard) says they are the same voice. Their time adds up;
   at 60 s together the speaker keeps the number, the spans held earlier
   become theirs, and there is no rewind. A recurring ad voice is a speaker
   like any other once it has a minute.
4. **The rewind.** The held speakers' frames are cut from the audio (0.3 s
   each side, pauses under 2 s filled), the diarizer goes back to the last
   state whose chunk, with its 3.2 s of context, ends before their first
   word, and hears the rest again without them. The rows from there are
   exactly those of a diarizer that never heard them. The cost is the
   seconds heard twice.
5. **A budget.** Seconds heard twice may not pass 0.25 of the recording so
   far plus 300 s. A rewind that would is skipped, and those speakers keep
   their numbers for good.

At the end, the speakers still held are played after the recording,
longest first, with 1 s of silence between, and take what numbers are
left. One that lands on no column (under 1 s, or under 20% of its
activity) is "other": its rows are no speaker's, and its words are no
numbered speaker's.

## The join check

Nemotron 3 sometimes gives a speaker a second number late in a recording:
a host keeps one number for 40 minutes and takes another for a later
stretch. A plain run does this too, but less where all eight numbers are
taken, because there is no spare to hand out. Freeing numbers lets it show.
(The cause is read from when the second number appears, not confirmed in
the model.)

At the end of the recording:

1. A diarizer that has heard nothing hears three 4 s clips of every speaker
   with 60 s or more alone, one speaker after another.
2. Two speakers whose clips it puts in one column are heard as a pair four
   times, on three other clips of each, twice in each order.
3. If it says one speaker every time, the later number joins the earlier.

Four trials, because one is not enough. With a single trial the check
joined four pairs the embedding model (below) called two voices. A person
listened to the four: two were one voice and two were two. Asked again and
again, the diarizer said one speaker in one to three of eight trials for a
pair the person heard as two, and every time for the pairs heard as one.

## Measured

2026-11-22, M4 Pro, libpktito 19ad40b, the int8 Nemotron 3 model; 100
episodes, 8 to 82 minutes each, decoded by ffmpeg 9.0.1. tools/shortspk/
rewind.py and core/hold.c both ran every episode.

**What happened.** 51 episodes reach seven numbers. In 50 of them
something was held, promoted or skipped: 88 holds, 6 promotions, and 49
rewinds skipped for the budget. Of 230 voices held, 196 took a speaker's
column at the end and 34 are "other".

**Cost**, as audio heard besides the recording's own, over its length:

| part                                 | median | mean | max   |
| ------------------------------------ | ------ | ---- | ----- |
| all of it, over the 100 episodes     | 3.9%   | 9.9  | 43.7% |
| all of it, the 50 where a rule acted | 16.5%  | 17.6 | 43.7% |
| the join check alone, over the 100   | 2.3%   | 2.5  | 12.2% |

In time, on the M4 Pro with the reference running beside it: 4 to 49 s an
episode, 28 s the median.

**The join check**, against a speaker-embedding model (ReDimNet2-B3, mean
cosine between the two speakers' clips, 0.40 or more taken as one voice;
tools/shortspk/merge.py): of 448 pairs of speakers with a minute alone, 43
are one voice. The first hearing puts 36 of them in one column, and 7 of
the 405 others. The four trials keep 35 of the 36, and 2 of the 7: one at
cosine 0.24 that a person heard as one voice, and one at 0.32 that no
person has heard.

**Who is right.** Where the plain run and this one disagree about two
stretches of a minute or more being one speaker, three listeners heard 4 s
clips of each: two speaker-embedding models (ReDimNet2-B3 and WeSpeaker's
ResNet34) and a Nemotron 3 that had heard nothing else. The two embedding
models agree on 1,013 of 1,014 pairs of stretches. Counting the pairs
where all three agree:

|                          | plain run | held and joined |
| ------------------------ | --------- | --------------- |
| two voices on one number | 13        | 0               |
| one voice on two numbers | 51        | 10              |
| wrong, in all            | 64        | 10              |

It is better than the plain run in 36 episodes, the same in 64 and worse in
none. One of the three listeners is the join check's own test, so this
score leans its way; the two embedding models had to agree for a pair to
count.

**The decoder matters.** The same mp3 decoded by ffmpeg 9.0.1 and 9.0.2
gives samples that differ, and the plain run with them: in one episode all
seven numbers are in use at 3.6 minutes with one decoder and at 10 with
the other. An earlier run of the rule on the other decode, with a rewind
that went back a step less far, scored 53 wrong for the plain run and 7
for this one. A recording's speakers are that decode's.

Not measured: any recording that is not a two-host podcast; the cost in
seconds on a small board; rewinding from the first short speaker and not
the seventh number, which in an earlier simulation made twice the new
splits for twice the cost.

## In the engine

- The high level only. `pk_diar_run` and `pk_speakers` are as they were;
  `pk_diar_run_held` leaves NeMo's answer on purpose, as
  `pk_asr_transcribe`'s retry does, and `--no-hold` is the switch.
- Memory: the rows twice over and one 0.6 MB state per step, 83 MB for an
  hour. The budget bounds how far back a rewind can go, so older states
  could be dropped; they are not yet.
- The audio is read again: where the diarizer goes back, and for the
  clips. pktito-c copies a pipe to a scratch file as it already did.
- A held speaker's words are still transcribed. One that lands on no
  speaker has no number: its rows are 0 in every column, and
  `pk_speakers` gives its words to whoever else is heard there, or no one.
- The constants are core/hold.c's, chosen by hand on four podcasts before
  the 100 were run; none was tuned on the 100.
