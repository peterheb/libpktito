"""A Parakeet TDT .nemo and Silero's silero_vad.onnx to one .pkf.

    pkito-tools convert parakeet-tdt-0.6b-v2.nemo silero_vad.onnx out.pkf

The numbers come from canon.py (the checkpoint's tensors under pkito's
names, with its exact folds) and silero.py (the VAD's, in vadito's
layout); this lays them out as core/pkf.h says and rounds the matrices:

  - every matrix of the encoder, the subsampling's pointwise convolutions,
    the prediction network and the joint is PK_Q8, int8 with a float32
    scale a row; the engine quantizes what they multiply to int8 too
    (README, "Why int8", has the measurements behind both);
  - everything else stays float32: the norms, the biases, the depthwise
    and 3x3 convolutions, the position biases, the embedding, the mel
    filterbank and window, and the VAD, which does not survive int8
    (vadito's README, "Why fp32").

Two layouts differ from NeMo's for the engine's sake, both permutations:
sub.out's inputs are frequency-major (pkito keeps the subsampling's
activations channel-last), and joint.out has zero rows to a multiple of 16.

The metadata says what the file was made from and what may be done with
it: the checkpoint's repo, file, commit and sha256, its licence (CC-BY-4.0,
whose attribution the notice carries), the VAD's source and licence (MIT),
and this converter's own stamp.
"""
import datetime
import hashlib
import os
import subprocess
import time
import unicodedata

import numpy as np

from . import __version__, canon, nemo, pkf, silero, spm

ARCH = "parakeet-tdt"


def punctuation(pieces):
    """NeMo's extract_punctuation_from_vocab (asr/parts/utils/
    tokenizer_utils.py): every character of category P* in a piece that is
    not special - bracketed, a ## continuation, starting with the space
    symbol, or blank. Its output is the set NeMo takes the space before away
    from in a transcript."""
    out = set()
    for p in pieces:
        if (p.startswith("[") and p.endswith("]")) or (p.startswith("<") and p.endswith(">")) \
                or p.startswith("##") or p.startswith(spm.SPACE) or not p.strip():
            continue
        out |= {c for c in p if unicodedata.category(c).startswith("P")}
    return "".join(sorted(out))


def front_matter(path):
    """The scalar `key: value` lines of a model card's YAML front matter."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "---":
            return out
        for line in f:
            if line.strip() == "---":
                break
            if line[:1].strip() and ":" in line:
                k, _, v = line.partition(":")
                v = v.strip().strip("\"'")
                if v:
                    out[k.strip()] = v
    return out


def card_languages(path):
    """The card's `language:` list, as `en,es,...`."""
    langs, on = [], False
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("language:"):
                    on = True
                    continue
                if on:
                    if line.startswith("- "):
                        langs.append(line[2:].strip())
                    else:
                        break
    return ",".join(langs)


def revision(path):
    """The HuggingFace commit a file was downloaded at, from the .metadata
    huggingface_hub leaves beside it, or ""."""
    d, base = os.path.split(path)
    md = os.path.join(d, ".cache", "huggingface", "download", base + ".metadata")
    try:
        with open(md, encoding="utf-8") as f:
            line = f.readline().strip()
    except OSError:
        return ""
    return line if len(line) == 40 and all(c in "0123456789abcdef" for c in line) else ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.decode().strip() if r.returncode == 0 else ""


def layout(canonical, a):
    """[(name, type, dims, array)] in file order: core/pkf.h's tensors."""
    C, D, M = a["sub_channels"], a["d_model"], a["n_mels"]
    V, ndur, N = a["vocab"], len(a["durations"]), a["n_fft"]
    out = []

    def f32(name, arr):
        arr = np.asarray(arr, np.float32)
        out.append((name, pkf.F32, arr.shape, arr))

    def q8(name, m):
        q, s = pkf.quantize_q8(m)
        out.append((name, pkf.Q8, q.shape, q))
        out.append((name + ".scale", pkf.F32, s.shape, s))

    k = np.arange(N // 2)
    for name, arr in canonical:
        if name == "mel.window":
            f32(name, arr)
            f32("mel.fft.cos", np.cos(2 * np.pi * k / N))
            f32("mel.fft.sin", np.sin(2 * np.pi * k / N))
        elif name == "sub.out.w":
            # NeMo's column c*F + f, F = M/8 frequencies, becomes f*C + c
            F = M // 8
            q8(name, arr.reshape(D, C, F).transpose(0, 2, 1).reshape(D, F * C))
        elif name == "joint.out.w" or name == "joint.out.b":
            rows = V + 1 + ndur
            pad = (-rows) % 16
            if arr.ndim == 2:
                q8(name, np.concatenate([arr, np.zeros((pad, arr.shape[1]))]))
            else:
                f32(name, np.concatenate([arr, np.zeros(pad)]))
        elif arr.ndim == 2 and name not in ("mel.fb", "dec.embed") and not name.endswith(
                (".conv0.w", ".dw1.w", ".dw2.w", "conv.dw")):
            q8(name, arr)
        else:
            f32(name, arr)
    return out


def convert(src, vad_src, dst, log=print):
    t0 = time.time()
    nm = nemo.Nemo(src)
    a = canon.arch(nm.config)
    sp = spm.Model(nm.tokenizer_model())
    if len(sp.pieces) != a["vocab"]:
        raise ValueError("the tokenizer has %d pieces and the decoder %d" % (len(sp.pieces), a["vocab"]))
    if 6 in sp.types:
        raise ValueError("the tokenizer has byte pieces, which pkito does not decode")
    log("  read %s: %d tensors" % (os.path.basename(src), len(nm.state)))
    canonical = canon.tensors(nm.state, a)
    tensors = layout(canonical, a)
    vad, vad_sha = silero.tensors(vad_src)
    tensors += [(n, pkf.F32, arr.shape, arr) for n, arr in vad]

    params = [(k, v) for k, v in a.items() if k != "durations"]
    params += [("n_durations", len(a["durations"]))]
    params += [("dur%d" % i, d) for i, d in enumerate(a["durations"])]

    name = os.path.splitext(os.path.basename(src))[0]
    card = os.path.join(os.path.dirname(src), "README.md")
    fm = front_matter(card)
    n_params = sum(int(np.prod(arr.shape)) for _, arr in canonical if not _.startswith(("mel.", "pos.")))
    meta = {"family": "Parakeet TDT",
            "params": "%.0fM" % (n_params / 1e6),
            "source": "nvidia/" + name,
            "source_file": os.path.basename(src)}
    rev = revision(src)
    if rev:
        meta["revision"] = rev
    meta["source_sha256"] = sha256_file(src)
    meta["card"] = "https://huggingface.co/nvidia/" + name
    if fm.get("license"):
        meta["license"] = fm["license"]
    meta["credit"] = "NVIDIA"
    meta["notice"] = ("%s by NVIDIA, https://huggingface.co/nvidia/%s, licensed CC-BY-4.0 "
                      "(creativecommons.org/licenses/by/4.0). This file holds its weights "
                      "rounded to int8 and rearranged by pkito-tools." % (name, name))
    langs = card_languages(card)
    if langs:
        meta["languages"] = langs
    meta["nemo_version"] = str(nm.config.get("nemo_version", ""))
    meta["vad"] = "Silero VAD (github.com/snakers4/silero-vad), MIT, Copyright (c) 2020-present Silero Team"
    meta["vad_source_file"] = os.path.basename(vad_src)
    meta["vad_sha256"] = vad_sha
    meta["recipe"] = ("convert: matrices q8 (int8, a scale a row); activations int8 in "
                      "blocks of 32 (encoder) or a row (decoder, joint)")
    sha = git_sha()
    meta["converter"] = "pkito-tools %s%s" % (__version__, " " + sha if sha else "")
    meta["converted"] = datetime.date.today().isoformat()
    meta["pkito_min"] = "0.1.0"

    vocab = pkf.vocab_blob(sp.pieces, sp.types, punctuation(sp.pieces))
    size = pkf.write(dst, ARCH, name, params, tensors, vocab, meta)
    log("  wrote %s: %d tensors, %.1f MB, in %.0f s" % (dst, len(tensors), size / 1e6, time.time() - t0))
    return dst
