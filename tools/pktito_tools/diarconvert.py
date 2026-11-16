"""NVIDIA's Nemotron 3 Diarization .nemo to a .pkf of its own.

    pktito-tools convert Nemotron-3-Diarization.nemo out.pkf

The numbers are nemotron3.py's (the reference's tensors() and arch(), in
the 30.4 s configuration its OFFLINE names); this lays them out as
core/pkf.h's "sortformer" architecture and rounds the matrices as convert.py
rounds a recognizer's: every matrix PK_Q8, int8 with a float32 scale a row,
and everything else float32 - the norms, the biases, the silence frame, the
mel filterbank and window (the checkpoint's own, Parakeet's rounded to
bfloat16), and the rotary positions' 32 frequencies. tools/diarquant.py
measured that rounding against float32 (docs/DIARIZATION.md, "Why int8").

Two layouts differ from NeMo's, both for the engine: the head's last
matrix, one row a speaker, has zero rows to 16 (q8's panels), and the
upsampler's convolution over three frames is one matrix whose columns are
the three frames side by side (nemotron3.tensors).

The rotary frequencies are NeMo's and Transformers' formula, 1 / theta^(i /
32) from a float32 arange, with the power computed in float64 and rounded
once and the reciprocal in float32 (torch's own float32 power may land an
ulp away), so that the engine takes no power.

The metadata carries what OpenMDW-1.1 asks of anyone who distributes the
model: a copy of the agreement (tools/licenses/OpenMDW-1.1.txt, whose
wording is openmdw.ai/license/1-1's, its quotation marks straightened) and
the model's notices of origin.
"""

import datetime
import os
import time

import numpy as np

from . import __version__, convert, nemo, nemotron3, pkf

ARCH = "sortformer"
LICENSE_TEXT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "licenses",
    "OpenMDW-1.1.txt",
)


def is_diarizer(cfg):
    """A Sortformer checkpoint's config, which this converts (a recognizer's
    is convert.py's)."""
    return str(cfg.get("target", "")).endswith("SortformerEncLabelModel")


def rope_freq(a):
    """The rotary positions' frequencies 1 / theta^(2i / dk), i < dk / 2: the
    power in float64 rounded once to float32, then the reciprocal in float32,
    as torch forms inv_freq from a float32 arange."""
    dk = a["d_model"] // a["n_heads"]
    x = np.arange(0, dk, 2, dtype=np.float32) / np.float32(dk)
    p = np.power(np.float64(a["rope_base"]), x.astype(np.float64)).astype(np.float32)
    return np.float32(1.0) / p


def params(a):
    """core/pkf.h's sortformer params, in file order."""
    return [
        ("sample_rate", a["sample_rate"]),
        ("n_fft", a["n_fft"]),
        ("win_length", a["win_length"]),
        ("hop_length", a["hop_length"]),
        ("n_mels", a["n_mels"]),
        ("preemph", float(a["preemph"])),
        ("log_guard", float(a["log_guard"])),
        ("n_layers", a["n_layers"]),
        ("d_model", a["d_model"]),
        ("n_heads", a["n_heads"]),
        ("ff_dim", a["ff_dim"]),
        ("ln_eps", float(a["ln_eps"])),
        ("rope_theta", float(a["rope_base"])),
        ("stack", a["subsampling"]),
        ("proj_dim", a["tf_d_model"]),
        ("upsample", a["upsample"]),
        ("n_spk", a["n_spk"]),
        ("chunk_len", int(a["chunk_len"])),
        ("chunk_left", int(a["chunk_left_context"])),
        ("chunk_right", int(a["chunk_right_context"])),
        ("fifo_len", int(a["fifo_len"])),
        ("cache_len", int(a["spkcache_len"])),
        ("update_period", int(a["spkcache_update_period"])),
        ("sil_frames", int(a["spkcache_sil_frames_per_spk"])),
        ("score_threshold", float(a["pred_score_threshold"])),
        ("boost_latest", float(a["scores_boost_latest"])),
        ("strong_boost", float(a["strong_boost_rate"])),
        ("weak_boost", float(a["weak_boost_rate"])),
        ("min_pos_scores", float(a["min_pos_scores_rate"])),
    ]


def layout(w, a):
    """[(name, type, dims, array)] in file order: core/pkf.h's sortformer
    tensors, from nemotron3.tensors' {name: float64 array}."""
    N, S = a["n_fft"], a["n_spk"]
    out = []

    def f32(name, arr):
        arr = np.asarray(arr, np.float32)
        out.append((name, pkf.F32, arr.shape, arr))

    def q8(name, m):
        q, s = pkf.quantize_q8(m)
        out.append((name, pkf.Q8, q.shape, q))
        out.append((name + ".scale", pkf.F32, s.shape, s))

    k = np.arange(N // 2)
    f32("mel.fb", w["mel.fb"])
    f32("mel.window", w["mel.window"])
    f32("mel.fft.cos", np.cos(2 * np.pi * k / N))
    f32("mel.fft.sin", np.sin(2 * np.pi * k / N))
    f32("rope.freq", rope_freq(a))
    q8("stack", w["stack"])
    f32("embed_norm.w", w["embed_norm.w"])
    f32("embed_norm.b", w["embed_norm.b"])
    for i in range(a["n_layers"]):
        q = "l%d." % i
        f32(q + "att.norm.w", w[q + "att.norm.w"])
        f32(q + "att.norm.b", w[q + "att.norm.b"])
        for m in ("att.q", "att.k", "att.v", "att.o"):
            q8(q + m, w[q + m])
        f32(q + "att.o.b", w[q + "att.o.b"])
        f32(q + "ff.norm.w", w[q + "ff.norm.w"])
        f32(q + "ff.norm.b", w[q + "ff.norm.b"])
        q8(q + "ff.up", w[q + "ff.up"])
        f32(q + "ff.up.b", w[q + "ff.up.b"])
        q8(q + "ff.down", w[q + "ff.down"])
        f32(q + "ff.down.b", w[q + "ff.down.b"])
    f32("final_norm.w", w["final_norm.w"])
    f32("final_norm.b", w["final_norm.b"])
    q8("proj", w["proj"])
    f32("proj.b", w["proj.b"])
    q8("up", w["up"])
    f32("up.b", w["up.b"])
    q8("head.hidden", w["head.hidden"])
    f32("head.hidden.b", w["head.hidden.b"])
    pad = (-S) % 16
    C = w["head.out"].shape[1]
    q8("head.out", np.concatenate([w["head.out"], np.zeros((pad, C))]))
    f32("head.out.b", np.concatenate([w["head.out.b"], np.zeros(pad)]))
    f32("sil", w["sil"])
    return out


def convert_file(src, dst, log=print):
    t0 = time.time()
    nm = nemo.Nemo(src)
    if not is_diarizer(nm.config):
        raise ValueError("%s is not a Sortformer diarizer's checkpoint" % src)
    a = nemotron3.arch(nm.config, nemotron3.OFFLINE)
    log("  read %s: %d tensors" % (os.path.basename(src), len(nm.state)))
    w = nemotron3.tensors(nm.state, a)
    tensors = layout(w, a)
    name = os.path.splitext(os.path.basename(src))[0]
    card = os.path.join(os.path.dirname(src), "README.md")
    fm = convert.front_matter(card)
    n_params = sum(
        int(np.prod(v.shape)) for k, v in w.items() if not k.startswith("mel.")
    )
    with open(LICENSE_TEXT, encoding="utf-8") as f:
        agreement = f.read()
    meta = {
        "family": "Nemotron 3 Diarization (Streaming Sortformer)",
        "params": "%.1fM" % (n_params / 1e6),
        "source": "nvidia/" + name,
        "source_file": os.path.basename(src),
    }
    rev = convert.revision(src)
    if rev:
        meta["revision"] = rev
    meta["source_sha256"] = convert.sha256_file(src)
    meta["card"] = "https://huggingface.co/nvidia/" + name
    meta["license"] = fm.get("license", "openmdw-1.1")
    meta["credit"] = "NVIDIA"
    meta["notice"] = (
        "%s by NVIDIA, https://huggingface.co/nvidia/%s, under the OpenMDW License "
        "Agreement, version 1.1 (openmdw.ai/license/1-1), whose text is license_text. "
        "This file holds its weights rounded to int8 and rearranged by pktito-tools."
        % (name, name)
    )
    meta["license_text"] = agreement
    meta["nemo_version"] = str(nm.config.get("nemo_version", ""))
    meta["configuration"] = (
        "streaming, 30.4 s: chunks of %d frames, %d of right context, a FIFO of %d, "
        "a speaker cache of %d updated every %d"
        % (
            a["chunk_len"],
            a["chunk_right_context"],
            a["fifo_len"],
            a["spkcache_len"],
            a["spkcache_update_period"],
        )
    )
    meta["recipe"] = (
        "convert: matrices q8 (int8, a scale a row); activations int8 in blocks of 32"
    )
    sha = convert.git_sha()
    meta["converter"] = "pktito-tools %s%s" % (__version__, " " + sha if sha else "")
    meta["converted"] = datetime.date.today().isoformat()
    size = pkf.write(dst, ARCH, name, params(a), tensors, b"", meta)
    log(
        "  wrote %s: %d tensors, %.1f MB, in %.0f s"
        % (dst, len(tensors), size / 1e6, time.time() - t0)
    )
    return dst
