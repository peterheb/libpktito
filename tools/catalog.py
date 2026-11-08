"""models.json as core/catalog.c, the table pktito-c reads (core/editions.h).

models.json catalogs the model files models.pktito.net serves, and the
hardware each edition of a model runs on. pktito-app builds its own
catalog from the same file. This checks the file against the rules both
read it by, and writes it out as C:

    python3 tools/catalog.py           write core/catalog.c
    python3 tools/catalog.py --check   change nothing; exit 1 if core/catalog.c
                                       isn't models.json, or models.json breaks a rule

The rules: every model has one CPU edition, its whole .pkf named after it,
and every other edition the model's slim file and an encoder, a Core ML
archive (.mlmodelc.aar) or a QNN graph (.qnn); an edition's id is its
model's once; a file is its bytes and a SHA-256; a QNN chip is five
numbers, with the systems it runs on. Which edition fits which device is
core/editions.c's to say, and tests/editioncheck's to check, against
tests/editions.tsv. Python's standard library and nothing else."""

import json
import pathlib
import re
import sys

TOP = pathlib.Path(__file__).resolve().parent.parent
SOURCE = TOP / "models.json"
OUT = TOP / "core" / "catalog.c"
SYSTEMS = {"android", "windows", "linux", "macos", "ios"}


def fail(why):
    sys.exit(f"models.json: {why}")


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


def check(d):
    if not d.get("base", "").startswith("https://") or not d["base"].endswith("/"):
        fail(f"base {d.get('base')!r}")
    for m in d["models"]:
        name = m["name"]
        ids = [e["id"] for e in m["editions"]]
        if len(set(ids)) != len(ids):
            fail(f"{name}: an edition id twice")
        if [e["runtime"]["kind"] for e in m["editions"]].count("cpu") != 1:
            fail(f"{name}: not one CPU edition")
        for e in m["editions"]:
            where, rt = f"{name} {e['id']}", e["runtime"]
            if not e.get("about"):
                fail(f"{where}: no about, the hardware it runs on in words")
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


def c_str(s):
    return (
        "NULL" if s is None else json.dumps(s)
    )  # ASCII-safe, and C reads JSON's escapes


def c_part(p):
    return f"{{ {c_str(p['file'])}, {p['bytes']}u, {c_str(p['sha256'])} }}"


def write(d):
    out = [
        "/* models.json as C, for pktito-c (core/editions.h): written by",
        " * tools/catalog.py, which make check-python holds this file to. Edit",
        " * models.json and run python3 tools/catalog.py, not this. */",
        '#include "editions.h"',
        "",
        "#include <stddef.h>",
        "",
        f"const char ed_base[] = {c_str(d['base'])};",
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
            fields = [
                f".id = {c_str(e['id'])}",
                f".about = {c_str(e['about'])}",
                f".kind = ED_{rt['kind'].upper()}",
            ]
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
    out.append("const ed_model ed_catalog[] = {")
    for i, m in enumerate(d["models"]):
        out.append(
            f"    {{ {c_str(m['name'])}, {c_str(m['about'])}, model{i}, {len(m['editions'])} }},"
        )
    out.append("};")
    out.append(f"const int ed_n_models = {len(d['models'])};")
    return "\n".join(out) + "\n"


def main():
    d = json.loads(SOURCE.read_text(encoding="utf-8"))
    check(d)
    c = write(d)
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != c:
            sys.exit(
                "core/catalog.c isn't models.json: python3 tools/catalog.py writes it"
            )
        print(
            f"catalog: models.json checked, {len(d['models'])} models, core/catalog.c current"
        )
    elif sys.argv[1:]:
        sys.exit(__doc__)
    else:
        OUT.write_text(c, encoding="utf-8")


main()
