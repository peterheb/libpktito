"""models.json as core/catalog.c, the table pktito-c reads (core/editions.h).

models.json catalogs the model files models.pktito.net serves, and the
hardware each edition of a model runs on. pktito-app builds its own
catalog from the same file. This checks the file against the rules both
read it by, and writes it out as C. It does the same for
tests/editions-rules.json, a made-up catalog that tests/editions-rules.tsv
holds choose()'s rules to:

    python3 tools/catalog.py           write core/catalog.c and tests/rules-catalog.c
    python3 tools/catalog.py --check   change nothing; exit 1 if either isn't its
                                       JSON, or a JSON breaks a rule

A model has:
- a name;
- its task: transcribe or diarize;
- its status: stable, offered by default, or preview, which pktito-c
  models --all lists and pktito-c runs only when it is named;
- what it needs: the lowest pktito that loads its files (absent: any);
- an about line;
- the languages a transcriber writes (absent: any);
- its licence and its source page;
- its editions. An edition may carry a status and needs of its own: less
  stable than its model's, or needing more, never the reverse.

The rules:
- A transcriber has one CPU edition, its whole .pkf named after it. Every
  other edition is the model's slim file and an encoder: a Core ML archive
  (.mlmodelc.aar) or a QNN graph (.qnn). A diarizer has the CPU edition
  alone.
- An edition's id appears in its model once.
- A file is its bytes and a SHA-256, the same wherever its name appears.
- A QNN chip is five numbers, with the systems it runs on.
- A language is one of core/langwords.h's.
- Where models/ holds one of a model's .pkf files, its metadata must agree:
  its licence and card with license and source, and its languages with
  languages.

Which edition fits which device is core/editions.c's to say, and
tests/editioncheck's to check, against tests/editions.tsv and
tests/editions-rules.tsv. Python's standard library and nothing else."""

import json
import pathlib
import re
import struct
import sys

TOP = pathlib.Path(__file__).resolve().parent.parent
CATALOGS = [  # (source, C file, the C names' prefix)
    (TOP / "models.json", TOP / "core" / "catalog.c", "ed_"),
    (
        TOP / "tests" / "editions-rules.json",
        TOP / "tests" / "rules-catalog.c",
        "ed_rules_",
    ),
]
MODELS = TOP / "models"
SYSTEMS = {"android", "windows", "linux", "macos", "ios"}
TASKS = ("transcribe", "diarize")
STATUSES = ("stable", "preview")
PKF_MAGIC = b"PKF\x01"


def version(v):
    """ "0.9.10" as (0, 9, 10), which orders as versions do"""
    return tuple(int(x) for x in v.split("."))


def languages_core():
    """the language codes core/ uses: core/langwords.h's PK_LANG_CODES"""
    h = (TOP / "core" / "langwords.h").read_text(encoding="utf-8")
    table = re.search(r"PK_LANG_CODES\[\] = \{([^}]*)\}", h)
    return set(re.findall(r'"([a-z]+)"', table.group(1)))


def pkf_meta(path):
    """a .pkf's metadata: core/pkf.h's header gives where its key\\0value\\0
    pairs are"""
    with open(path, "rb") as f:
        h = f.read(256)
        if len(h) < 160 or h[:4] != PKF_MAGIC:
            return None
        off, size = struct.unpack_from("<QQ", h, 144)
        f.seek(off)
        raw = f.read(size)
    p = raw.split(b"\0")
    return {p[i].decode(): p[i + 1].decode() for i in range(0, len(p) - 1, 2)}


def check(d, src, langs):
    def fail(why):
        sys.exit(f"{src}: {why}")

    def part(p, where, suffix):
        if set(p) - {"file", "bytes", "sha256", "made"} or not {
            "file",
            "bytes",
            "sha256",
        } <= set(p):
            fail(f"{where}: a file is file, bytes, sha256 and made")
        if not isinstance(p["bytes"], int) or p["bytes"] <= 0:
            fail(f"{where}: bytes {p['bytes']!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", p["sha256"]):
            fail(f"{where}: sha256 {p['sha256']!r}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", p["file"]) or not p["file"].endswith(
            suffix
        ):
            fail(f"{where}: {p['file']!r} isn't a {suffix} file")

    def needs(v, where):
        if v is not None and not (
            isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+", v)
        ):
            fail(f"{where}: needs {v!r} isn't a version, major.minor.patch")

    if not d.get("base", "").startswith("https://") or not d["base"].endswith("/"):
        fail(f"base {d.get('base')!r}")
    files = {}  # a file's name is one file wherever it appears
    for m in d["models"]:
        for e in m["editions"]:
            for p in (e.get("model"), e.get("encoder")):
                if p and files.setdefault(p["file"], (p["bytes"], p["sha256"])) != (
                    p["bytes"],
                    p["sha256"],
                ):
                    fail(
                        f"{p['file']}: two files of one name, by their bytes or SHA-256"
                    )
    names = [m["name"] for m in d["models"]]
    if len(set(names)) != len(names):
        fail("a model's name twice")
    for m in d["models"]:
        name = m["name"]
        if not re.fullmatch(r"[a-z0-9.-]+", name):
            fail(f"{name!r}: a name is [a-z0-9.-]")
        if m.get("task") not in TASKS:
            fail(f"{name}: task {m.get('task')!r}, not one of {TASKS}")
        if m.get("status") not in STATUSES:
            fail(f"{name}: status {m.get('status')!r}, not one of {STATUSES}")
        needs(m.get("needs"), name)
        if not m.get("about"):
            fail(f"{name}: no about")
        if "languages" in m and (not m["languages"] or set(m["languages"]) - langs):
            fail(f"{name}: languages {m['languages']!r}, not core/langwords.h's")
        if m["task"] == "diarize" and "languages" in m:
            fail(f"{name}: a diarizer hears voices, not words: no languages")
        if not re.fullmatch(r"[A-Za-z0-9.-]+", m.get("license", "")):
            fail(f"{name}: license {m.get('license')!r}, a licence's short name")
        if not m.get("source", "").startswith("https://"):
            fail(f"{name}: source {m.get('source')!r}, the checkpoint's page")
        ids = [e["id"] for e in m["editions"]]
        if len(set(ids)) != len(ids):
            fail(f"{name}: an edition id twice")
        kinds = [e["runtime"]["kind"] for e in m["editions"]]
        if kinds.count("cpu") != 1:
            fail(f"{name}: not one CPU edition")
        if m["task"] == "diarize" and kinds != ["cpu"]:
            fail(f"{name}: a diarizer's one edition is the CPU's")
        for e in m["editions"]:
            where, rt = f"{name} {e['id']}", e["runtime"]
            if not e.get("about"):
                fail(f"{where}: no about, the hardware it runs on in words")
            if e.get("status", m["status"]) not in STATUSES:
                fail(f"{where}: status {e['status']!r}")
            if m["status"] == "preview" and e.get("status") == "stable":
                fail(f"{where}: an edition stabler than its model")
            needs(e.get("needs"), where)
            if (
                m.get("needs")
                and e.get("needs")
                and version(e["needs"]) < version(m["needs"])
            ):
                fail(f"{where}: an edition needing less than its model")
            if rt["kind"] == "cpu":
                if e["id"] != "cpu" or "encoder" in e:
                    fail(f"{where}: the CPU edition is id cpu, with no encoder")
                part(e["model"], where, ".pkf")
                if e["model"]["file"] != f"{name}.pkf":
                    fail(f"{where}: the whole file is {name}.pkf")
                continue
            part(e["model"], where, ".pkf")
            if e["model"]["file"] != f"{name}-slim.pkf":
                fail(f"{where}: an NPU edition loads {name}-slim.pkf")
            if rt["kind"] == "coreml":
                floors = (rt.get("macos"), rt.get("ios"))
                if not all(isinstance(f, list) and len(f) == 2 for f in floors):
                    fail(f"{where}: Core ML's macos and ios floors are [major, minor]")
                part(e["encoder"], where, ".mlmodelc.aar")
            elif rt["kind"] == "qnn":
                if not re.fullmatch(r"\d+(,\d+){4}", rt.get("chip", "")):
                    fail(f"{where}: a QNN chip is five numbers, not {rt.get('chip')!r}")
                if not rt.get("systems") or set(rt["systems"]) - SYSTEMS:
                    fail(f"{where}: systems {rt.get('systems')!r}")
                part(e["encoder"], where, ".qnn")
            else:
                fail(f"{where}: runtime {rt['kind']!r}")


def check_files(d, src):
    """Each of the catalog's .pkf files models/ holds, against its entry:
    the files it checked"""
    seen = []
    for m in d["models"]:
        for e in m["editions"]:
            path = MODELS / e["model"]["file"]
            if e["model"]["file"] in seen or not path.is_file():
                continue
            meta, rel = pkf_meta(path), path.relative_to(TOP).as_posix()
            if meta is None:
                sys.exit(f"{src}: {rel} isn't a .pkf")
            where = f"{src}: {m['name']}, {rel}"
            if meta.get("license", "").lower() != m["license"].lower():
                sys.exit(
                    f"{where}: its licence {meta.get('license')!r}, not {m['license']!r}"
                )
            if meta.get("card") != m["source"]:
                sys.exit(f"{where}: its card {meta.get('card')!r}, not {m['source']!r}")
            have = set(meta["languages"].split(",")) if meta.get("languages") else None
            want = set(m["languages"]) if "languages" in m else None
            if have != want:
                sys.exit(
                    f"{where}: its languages {meta.get('languages')!r}, not {m.get('languages')!r}"
                )
            seen.append(e["model"]["file"])
    return seen


def c_str(s):
    return (
        "NULL" if s is None else json.dumps(s)
    )  # ASCII-safe, and C reads JSON's escapes


def c_part(p):
    return f"{{ {c_str(p['file'])}, {p['bytes']}u, {c_str(p['sha256'])} }}"


def write(d, src, prefix):
    out = [
        f"/* {src} as C, for pktito-c (core/editions.h): written by",
        " * tools/catalog.py, which make check-python holds this file to. Edit",
        f" * {src} and run python3 tools/catalog.py, not this. */",
        '#include "editions.h"',
        "",
        "#include <stddef.h>",
        "",
        f"const char {prefix}base[] = {c_str(d['base'])};",
        "",
    ]
    systems = sorted(
        {
            tuple(e["runtime"]["systems"])
            for m in d["models"]
            for e in m["editions"]
            if "systems" in e["runtime"]
        }
    )
    for s in systems:
        out.append(
            f"static const char *const on_{'_'.join(s)}[] = {{ {', '.join(c_str(x) for x in s)}, NULL }};"
        )
    out.append("")
    for i, m in enumerate(d["models"]):
        out.append(f"static const ed_edition model{i}[] = {{")
        for e in m["editions"]:
            rt = e["runtime"]
            # the edition's own status (core/editions.c weighs its model's
            # too), and its needs, else its model's
            status = e.get("status", "stable")
            needs = e.get("needs", m.get("needs"))
            fields = [
                f".id = {c_str(e['id'])}",
                f".about = {c_str(e['about'])}",
                f".kind = ED_{rt['kind'].upper()}",
                f".status = ED_{status.upper()}",
            ]
            if needs:
                fields.append(f".needs = {c_str(needs)}")
            if rt["kind"] == "coreml":
                fields += [
                    f".macos = {{ {rt['macos'][0]}, {rt['macos'][1]} }}",
                    f".ios = {{ {rt['ios'][0]}, {rt['ios'][1]} }}",
                ]
            if rt["kind"] == "qnn":
                fields += [
                    f".chip = {c_str(rt['chip'])}",
                    f".systems = on_{'_'.join(rt['systems'])}",
                ]
            fields.append(f".model = {c_part(e['model'])}")
            if "encoder" in e:
                fields.append(f".encoder = {c_part(e['encoder'])}")
                note = e["encoder"].get("made", {}).get("note")
                if note:
                    fields.append(f".note = {c_str(note)}")
            out.append("    {")
            out += [f"        {f}," for f in fields]
            out.append("    },")
        out.append("};")
        out.append("")
    out.append(f"const ed_model {prefix}catalog[] = {{")
    for i, m in enumerate(d["models"]):
        out.append("    {")
        out += [
            f"        .name = {c_str(m['name'])},",
            f"        .about = {c_str(m['about'])},",
            f"        .task = ED_{m['task'].upper()},",
            f"        .status = ED_{m['status'].upper()},",
            f"        .needs = {c_str(m.get('needs'))},",
            f"        .editions = model{i},",
            f"        .n_editions = {len(m['editions'])},",
        ]
        out.append("    },")
    out.append("};")
    out.append(f"const int {prefix}n_models = {len(d['models'])};")
    return "\n".join(out) + "\n"


def main():
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(__doc__)
    langs = languages_core()
    said = []
    for source, out, prefix in CATALOGS:
        src = source.relative_to(TOP).as_posix()
        d = json.loads(source.read_text(encoding="utf-8"))
        check(d, src, langs)
        files = check_files(d, src) if prefix == "ed_" else []
        c = write(d, src, prefix)
        if sys.argv[1:] == ["--check"]:
            if not out.exists() or out.read_text(encoding="utf-8") != c:
                sys.exit(
                    f"{out.relative_to(TOP).as_posix()} isn't {src}: python3 tools/catalog.py writes it"
                )
            said.append(
                f"{src} checked, {len(d['models'])} models"
                + (
                    f", the credit lines of {len(files)} .pkf files in models/"
                    if files
                    else ""
                )
            )
        else:
            out.write_text(c, encoding="utf-8")
    if said:
        print("catalog: " + "; ".join(said) + "; the C current")


main()
