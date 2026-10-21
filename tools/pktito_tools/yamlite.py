"""The subset of YAML a NeMo model_config.yaml is written in, read without
PyYAML: block mappings and block sequences by indentation, plain and quoted
scalars, flow `[]` and `{}` only when empty. Anything else is refused, not
guessed at; the converter reads a few dozen keys out of the result and would
rather stop than read a wrong one.
"""
import re

_INT = re.compile(r"^[-+]?[0-9]+$")
_FLOAT = re.compile(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$")


def _scalar(s):
    s = s.strip()
    if s in ("", "~", "null", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if s == "[]":
        return []
    if s == "{}":
        return {}
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return bytes(s[1:-1], "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    if s[0] in "[{&*!|>%@`":
        raise ValueError("unsupported YAML scalar: %r" % s)
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s) or s in (".inf", "-.inf", ".nan"):
        return float(s.replace(".inf", "inf").replace(".nan", "nan"))
    return s


def _split_key(line):
    """`key: value` -> (key, value text), or None if the line is not one. A
    colon inside quotes is not the separator."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == ":" and (i + 1 == len(line) or line[i + 1] == " "):
            k = line[:i].strip()
            if len(k) >= 2 and k[0] == k[-1] and k[0] in "'\"":
                k = k[1:-1]
            return k, line[i + 1:]
    return None


def loads(text):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise ValueError("tab indentation")
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    pos = [0]

    def block(indent):
        if pos[0] >= len(lines):
            return None
        first = lines[pos[0]][1]
        if first.startswith("- ") or first == "-":
            out = []
            while pos[0] < len(lines) and lines[pos[0]][0] == indent and \
                    (lines[pos[0]][1].startswith("- ") or lines[pos[0]][1] == "-"):
                item = lines[pos[0]][1][1:].strip()
                pos[0] += 1
                kv = _split_key(item) if item else None
                if kv is None:
                    if item:
                        out.append(_scalar(item))
                    else:
                        out.append(block(lines[pos[0]][0]) if pos[0] < len(lines)
                                   and lines[pos[0]][0] > indent else None)
                else:
                    # "- key: value" opens a mapping whose further keys sit
                    # two columns in, under the key
                    d = {}
                    k, v = kv
                    if v.strip():
                        d[k] = _scalar(v)
                    elif pos[0] < len(lines) and lines[pos[0]][0] > indent + 2:
                        d[k] = block(lines[pos[0]][0])
                    else:
                        d[k] = None
                    while pos[0] < len(lines) and lines[pos[0]][0] == indent + 2:
                        sub = block(indent + 2)
                        if not isinstance(sub, dict):
                            raise ValueError("mixed sequence item")
                        d.update(sub)
                    out.append(d)
            return out
        out = {}
        while pos[0] < len(lines) and lines[pos[0]][0] == indent:
            line = lines[pos[0]][1]
            if line.startswith("- "):
                break
            kv = _split_key(line)
            if kv is None:
                raise ValueError("not a mapping line: %r" % line)
            k, v = kv
            pos[0] += 1
            if v.strip():
                out[k] = _scalar(v)
            elif pos[0] < len(lines) and (lines[pos[0]][0] > indent or (
                    lines[pos[0]][0] == indent and lines[pos[0]][1].startswith("- "))):
                out[k] = block(lines[pos[0]][0])
            else:
                out[k] = None
        return out

    doc = block(0)
    if pos[0] != len(lines):
        raise ValueError("unparsed YAML from line %r" % (lines[pos[0]][1],))
    return doc
