"""pktito-tools convert | tiny | info | ref | diar"""

import argparse
import sys

import numpy as np

from . import __version__


def read_wav(path):
    """Mono float64 at 16 kHz (wavio.py)."""
    from . import wavio

    a, rate = wavio.read(path)
    if rate != 16000:
        sys.exit("%s: %d Hz; the reference reads 16000 only" % (path, rate))
    return a


def pkf_weights(path):
    """A .pkf's model as the reference takes it: canonical names, float64,
    q8 matrices dequantized, and NeMo's layouts restored."""
    from . import pkf

    f = pkf.File(path)
    a = dict(f.params)
    a["durations"] = [f.params["dur%d" % i] for i in range(f.params["n_durations"])]
    w = {}
    for name in f.descs:
        if (
            name.endswith(".scale")
            or name.startswith("vad.")
            or name.startswith("mel.fft")
        ):
            continue
        w[name] = f.tensor(name).astype(np.float64)
    C, D, F = a["sub_channels"], a["d_model"], a["n_mels"] // 8
    w["sub.out.w"] = (
        w["sub.out.w"].reshape(D, F, C).transpose(0, 2, 1).reshape(D, C * F)
    )
    n_out = a["vocab"] + 1 + a["n_durations"]
    w["joint.out.w"], w["joint.out.b"] = (
        w["joint.out.w"][:n_out],
        w["joint.out.b"][:n_out],
    )
    return w, a, f


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pktito-tools")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "convert",
        help="model.nemo silero_vad.onnx -> .pkf (a recognizer), or "
        "Nemotron-3-Diarization.nemo -> .pkf (the diarizer)",
    )
    c.add_argument("files", nargs="+", metavar="FILE")
    t = sub.add_parser("tiny", help="a tiny .pkf with random weights, for the C checks")
    t.add_argument(
        "--diarizer", action="store_true", help="a diarizer's, not a recognizer's"
    )
    t.add_argument("dst")
    i = sub.add_parser("info", help="a .pkf's metadata, params and tensors")
    i.add_argument("file")
    r = sub.add_parser("ref", help="transcribe a 16 kHz WAV with the float64 reference")
    r.add_argument(
        "model", help="a .nemo (the checkpoint itself) or a .pkf (its rounded weights)"
    )
    r.add_argument("wav", nargs="+")
    d = sub.add_parser(
        "diar",
        help="who speaks when in a 16 kHz WAV, as RTTM, with the float64 "
        "Streaming Sortformer reference (docs/DIARIZATION.md)",
    )
    d.add_argument(
        "model",
        help="Nemotron-3-Diarization.nemo or diar_streaming_sortformer_4spk-v2.nemo",
    )
    d.add_argument("wav", nargs="+")
    a = ap.parse_args(argv)

    if a.cmd == "convert":
        from . import convert, diarconvert

        if len(a.files) == 2:
            diarconvert.convert_file(a.files[0], a.files[1])
        elif len(a.files) == 3:
            convert.convert(*a.files)
        else:
            sys.exit(
                "pktito-tools convert: model.nemo silero_vad.onnx out.pkf, or "
                "Nemotron-3-Diarization.nemo out.pkf"
            )
    elif a.cmd == "tiny":
        from . import tiny

        size = (tiny.write_diarizer if a.diarizer else tiny.write)(a.dst)
        print("%s: %d bytes" % (a.dst, size))
    elif a.cmd == "info":
        from . import pkf

        f = pkf.File(a.file)
        print(
            "%s: %s (%s), version %d, digest %s"
            % (a.file, f.name, f.arch, f.version, f.digest)
        )
        for k, v in f.meta.items():
            print("  %-16s %s" % (k, v))
        print("params")
        for k, v in f.params.items():
            print("  %-16s %s" % (k, v))
        total = 0
        for k, (typ, ndim, dims, off, nbytes) in f.descs.items():
            total += nbytes
        print(
            "%d tensors, %.1f MB; vocabulary of %d, punctuation %r"
            % (len(f.descs), total / 1e6, len(f.pieces), f.punct)
        )
    elif a.cmd == "ref":
        from . import reference, spm

        if a.model.endswith(".pkf"):
            w, arch, f = pkf_weights(a.model)
            m = spm.Model.__new__(spm.Model)
            m.pieces, m.types = f.pieces, f.types
            m.unk_surface, m.add_dummy_prefix, m.remove_extra_whitespaces = (
                " ⁇ ",
                True,
                False,
            )
            punct = f.punct
        else:
            from . import canon, convert, nemo

            nm = nemo.Nemo(a.model)
            arch = canon.arch(nm.config)
            w = dict(canon.tensors(nm.state, arch))
            m = spm.Model(nm.tokenizer_model())
            punct = convert.punctuation(m.pieces)
        for path in a.wav:
            toks, _, _ = reference.transcribe(read_wav(path), w, arch)
            text = m.decode([k for k, _, _ in toks])
            for p in punct:
                text = text.replace(" " + p, p)
            print(text)
    elif a.cmd == "diar":
        import os
        from . import nemo, nemotron3, sortformer

        nm = nemo.Nemo(a.model)
        # Nemotron 3's encoder is NeMo's TransformerEncoder, v2's a ConformerEncoder
        rotary = (
            nm.config["encoder"]
            .get("_target_", "")
            .endswith("modules.TransformerEncoder")
        )
        net = nemotron3 if rotary else sortformer
        arch = net.arch(nm.config, net.OFFLINE)
        w = net.tensors(nm.state, arch)
        for path in a.wav:
            p, _ = net.diarize(read_wav(path), w, arch)
            name = os.path.splitext(os.path.basename(path))[0]
            for start, end, s in sortformer.segments(p, arch["frame"]):
                print(
                    "SPEAKER %s 1 %.2f %.2f <NA> <NA> speaker_%d <NA> <NA>"
                    % (name, start, end - start, s)
                )
    return 0
