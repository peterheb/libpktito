# tests/check-c.sh's commands for Windows without MSYS2, which has no make
# or sh: a build made elsewhere and copied over (Windows' own tar.exe, from
# Windows 10 1803, unpacks a tar of the tree). Run from anywhere with the
# programs built (make check-c's prerequisites: pktito-c, the Makefile's
# TESTS and examples/transcribe, whose transcript is the last check),
#
#   powershell -File tests\check-c.ps1 [-V2 v2.pkf] [-V3 v3.pkf] [-Model m.pkf]
#
# with -ExecutionPolicy RemoteSigned before -File where the policy is
# Restricted (Windows 10's and 11's default), which runs no script. -V2 and
# -V3 are the models test.wav's transcripts are held with, models\'s by
# default, each skipped with a line when it is not there; -Model is the one
# kcheck and the other test.wav checks use, -V2 by default. A relative path
# is the tree's, as check-c.sh takes MODEL. The first failure ends the run.
#
# A program's output goes to a file through cmd's > and is compared by its
# hash: PowerShell's own > and | decode it in the console's code page and
# write it out again, and that file is not the program's bytes.
param([string]$V2 = 'models\parakeet-tdt-0.6b-v2.pkf',
      [string]$V3 = 'models\parakeet-tdt-0.6b-v3.pkf',
      [string]$Model = $V2)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot)
$tiny = 'tests\data\tiny.pkf'

function run([string]$cmd) {
    $cmd
    cmd /c $cmd
    if ($LASTEXITCODE) { "FAIL ($LASTEXITCODE): $cmd"; exit 1 }
}
function same([string]$a, [string]$b) { (Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash }

run 'tests\mathcheck.exe'
run "tests\kcheck.exe `"$Model`""
run 'tests\q8check.exe'
run "tests\pkfcheck.exe $tiny tests\pkfcheck.wav"
run 'tests\wavcheck.exe tests\wavcheck.tmp'
run "tests\pathcheck.exe $tiny"
run 'tests\capcheck.exe'
run 'tests\langcheck.exe'
run 'tests\editioncheck.exe tests\editions.tsv'
run 'tests\editioncheck.exe tests\editions-rules.tsv'
run 'tests\chunkcheck.exe'
run "tests\transcheck.exe `"$Model`" tests\data\test.wav"
run "tests\sentcheck.exe `"$Model`" tests\data\test.wav"

cmd /c "pktito-c.exe -m $tiny --json tests\pkfcheck.wav > tests\pkfcheck.json"
$r = $LASTEXITCODE
foreach ($a in '--srt tests\pkfcheck.wav', 'tests\pkfcheck.wav tests\pkfcheck.wav', '--fast tests\pkfcheck.wav') {
    if (-not $r) { cmd /c "pktito-c.exe -m $tiny $a > NUL"; $r = $LASTEXITCODE }
}
if ($r) { 'pktito-c: FAIL on the tiny model'; exit 1 }
'pktito-c: the tiny model through the CLI, chunked, four ways'

# check-c.sh's names past ASCII, made from their code points so that this
# file means the same in every code page: Windows PowerShell 5.1 reads a
# script without a byte order mark in the system's.
$e, $n, $ya = [char]0xE9, [char]0xF1, [char]0x44F
$d = "tests\pathcheck-t${e}st-$n-$ya"
New-Item -ItemType Directory -Force $d | Out-Null
Copy-Item $tiny "$d\tiny-$e.pkf"
Copy-Item tests\pkfcheck.wav "$d\$n.wav"
cmd /c "pktito-c.exe -m `"$d\tiny-$e.pkf`" --json `"$d\$n.wav`" > tests\pathcheck.json"
$ok = (-not $LASTEXITCODE) -and (same tests\pathcheck.json tests\pkfcheck.json)
# and the model through the environment, which getenv gives in the ANSI
# code page, as main's argv
$was = $env:PKTITO_MODEL
$env:PKTITO_MODEL = "$d\tiny-$e.pkf"
cmd /c "pktito-c.exe --json `"$d\$n.wav`" > tests\pathcheck.json"
$ok = $ok -and (-not $LASTEXITCODE) -and (same tests\pathcheck.json tests\pkfcheck.json)
$env:PKTITO_MODEL = $was
Remove-Item -Recurse -Force $d, tests\pathcheck.json
if (-not $ok) { 'pktito-c: FAIL on a model and a WAV named past ASCII'; exit 1 }
'pktito-c: a model and a WAV named past ASCII, from its command line and $PKTITO_MODEL'
Remove-Item tests\pkfcheck.wav, tests\pkfcheck.json

cmd /c "pktito-c.exe probs -m `"$Model`" tests\data\test.wav > tests\probs.out"
if (-not (same tests\probs.out tests\data\test.probs.txt)) {
    'test.wav: FAIL - VAD probabilities differ from tests\data\test.probs.txt'; exit 1
}
"test.wav: VAD probabilities are vadito's, bit for bit"
Remove-Item tests\probs.out
foreach ($v in @(@('v2', $V2), @('v3', $V3))) {
    $name, $f = $v
    if (-not $f -or -not (Test-Path $f)) { "test.wav: no $f - skipped"; continue }
    cmd /c "pktito-c.exe --json -m `"$f`" tests\data\test.wav > tests\t.json"
    if (-not (same tests\t.json "tests\data\test.$name.json")) {
        "test.wav: FAIL - $name transcript differs from tests\data\test.$name.json"; exit 1
    }
    "test.wav: $name transcript is ref's, every token and time"
    cmd /c "pktito-c.exe --fast --json -m `"$f`" tests\data\test.wav > tests\t.json"
    if (-not (same tests\t.json "tests\data\test.$name.fast.json")) {
        "test.wav: FAIL - $name --fast transcript differs from tests\data\test.$name.fast.json"; exit 1
    }
    "test.wav: $name transcript in the fast mode is ref's"
    Remove-Item tests\t.json
}

# check-c.sh's --stream at 8 and 48 kHz, against the same samples resampled first
$s = 'tests\streamcheck'
foreach ($hz in 8000, 48000) {
    cmd /c "tests\streamcheck.exe tests\data\test.wav $hz $s-in.wav $s-16k.wav"
    $ok = -not $LASTEXITCODE
    if ($ok) { cmd /c "pktito-c.exe --stream --json -m `"$Model`" $s-in.wav > $s-in.json"; $ok = -not $LASTEXITCODE }
    if ($ok) { cmd /c "pktito-c.exe --stream --json -m `"$Model`" $s-16k.wav > $s-16k.json"; $ok = -not $LASTEXITCODE }
    $ok = $ok -and ((Get-Item "$s-in.json").Length -gt 0) -and (same "$s-in.json" "$s-16k.json")
    if (-not $ok) { "pktito-c --stream: FAIL at $hz Hz - not the lines of the same samples resampled first"; exit 1 }
    "pktito-c --stream: test.wav at $hz Hz, resampled as it streams, as resampled first"
}
Remove-Item "$s-in.wav", "$s-16k.wav", "$s-in.json", "$s-16k.json"

# Line by line, as PowerShell gives a program's output, without the line
# ends (examples/transcribe writes the C runtime's CR LF) and less the two
# times it puts before a chunk's text, then joined: the example writes a
# chunk a line, the CLI a sentence. -cne, since -ne ignores case.
$ex = cmd /c "examples\transcribe.exe `"$Model`" tests\data\test.wav" | ForEach-Object { ($_ -split ' ', 3)[2] }
$cli = cmd /c "pktito-c.exe --no-context --no-retry -m `"$Model`" tests\data\test.wav"
if ((($ex -join ' ') -replace ' +', ' ') -cne (($cli -join ' ') -replace ' +', ' ')) { "examples/transcribe: FAIL - not the CLI's transcript"; exit 1 }
"examples/transcribe: the CLI's transcript without the context and the retry, through the library alone"
'check-c: all passed'
