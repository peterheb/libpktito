"""A Parakeet TDT checkpoint in HuggingFace's transformers layout
(ParakeetForTDT: config.json, model.safetensors, tokenizer.json), read with
the standard library and numpy, and presented as nemo.Nemo presents a
.nemo: a NeMo config, a state dict under NeMo's names, the tokenizer.

transformers' layout keeps the network and drops what NeMo builds at load
time: the config's preprocessor and decoding sections, the mel filterbank
and window (preprocessor.featurizer.fb and .window, buffers NeMo computes
and saves), and the SentencePiece model. Those come from a base .nemo, the
checkpoint the HF one was made from, after checking that the two are the
same network: the HF config is the base's own HF config, field for field,
every tensor the base has is here under its name with its shape, and the
tokenizer is the base's. moondream/parakeet-ultra is
nvidia/parakeet-tdt-0.6b-v3 post-trained, its config.json and
tokenizer.json byte for byte v3's HF ones (2026-10-31), so its base is
parakeet-tdt-0.6b-v3.nemo.

NAMES maps every transformers name to NeMo's. The map is checked on the
one checkpoint that exists in both layouts, nvidia/parakeet-tdt-0.6b-v3's
model.safetensors against its .nemo: every tensor equal, bit for bit
(tests/hfcheck.py). Tensors past the network (moondream's vad_head, a
voice-activity head on the subsampler's output that nothing in the
transcript reads) are listed in `extra` and left out of the state.
"""

import json
import os
import re
import struct

import numpy as np

from . import nemo

# transformers' name -> NeMo's; \1 is the layer
NAMES = [
    (
        r"encoder\.subsampling\.layers\.(\d+)\.(weight|bias)",
        r"encoder.pre_encode.conv.\1.\2",
    ),
    (r"encoder\.subsampling\.linear\.(weight|bias)", r"encoder.pre_encode.out.\1"),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight",
        r"encoder.layers.\1.self_attn.linear_q.weight",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight",
        r"encoder.layers.\1.self_attn.linear_k.weight",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        r"encoder.layers.\1.self_attn.linear_v.weight",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight",
        r"encoder.layers.\1.self_attn.linear_out.weight",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.weight",
        r"encoder.layers.\1.self_attn.linear_pos.weight",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.bias_u",
        r"encoder.layers.\1.self_attn.pos_bias_u",
    ),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.bias_v",
        r"encoder.layers.\1.self_attn.pos_bias_v",
    ),
    (
        r"encoder\.layers\.(\d+)\.conv\.norm\.(.+)",
        r"encoder.layers.\1.conv.batch_norm.\2",
    ),
    (r"encoder\.layers\.(\d+)\.(.+)", r"encoder.layers.\1.\2"),
    (r"decoder\.embedding\.weight", r"decoder.prediction.embed.weight"),
    (r"decoder\.lstm\.(.+)", r"decoder.prediction.dec_rnn.lstm.\1"),
    (r"decoder\.decoder_projector\.(weight|bias)", r"joint.pred.\1"),
    (r"encoder_projector\.(weight|bias)", r"joint.enc.\1"),
    (r"joint\.head\.(weight|bias)", r"joint.joint_net.2.\1"),
]

# the base's tensors transformers does not carry
FROM_BASE = ("preprocessor.featurizer.fb", "preprocessor.featurizer.window")

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
}


def nemo_name(name):
    """NeMo's name for a transformers one, or None for one past the network."""
    for pat, rep in NAMES:
        if re.fullmatch(pat, name):
            return re.sub(pat, rep, name)
    return None


def safetensors(path):
    """{name: numpy array} of a .safetensors file: an 8-byte little-endian
    header length, a JSON header of each tensor's dtype, shape and byte
    range, then the bytes."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
        head.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for name, t in head.items():
            if t["dtype"] not in _DTYPES:
                raise ValueError(
                    "%s: %s is %s, which this reader does not take"
                    % (path, name, t["dtype"])
                )
            lo, hi = t["data_offsets"]
            dt = np.dtype(_DTYPES[t["dtype"]]).newbyteorder("<")
            if hi - lo != dt.itemsize * int(np.prod(t["shape"], dtype=np.int64)):
                raise ValueError("%s: %s's bytes are not its shape's" % (path, name))
            f.seek(base + lo)
            out[name] = np.frombuffer(f.read(hi - lo), dt).reshape(t["shape"])
        return out


class HF:
    def __init__(self, path, base):
        """path: the model.safetensors (config.json and tokenizer.json
        beside it); base: the .nemo it was made from."""
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        self.base = nemo.Nemo(base)
        self.config = self.base.config
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            self.hf_config = json.load(f)
        self._check_config()
        self._state = None
        self.extra = []

    def _check_config(self):
        """The HF config says the base's network: the numbers canon.arch
        reads from the base's NeMo config, as transformers names them."""
        h, e, cfg = (
            self.hf_config,
            self.hf_config.get("encoder_config", {}),
            self.config,
        )
        enc, pre = cfg["encoder"], cfg["preprocessor"]
        want = [
            (h.get("model_type"), "parakeet_tdt"),
            (e.get("hidden_size"), enc["d_model"]),
            (e.get("num_hidden_layers"), enc["n_layers"]),
            (e.get("num_attention_heads"), enc["n_heads"]),
            (e.get("intermediate_size"), enc["d_model"] * enc["ff_expansion_factor"]),
            (e.get("conv_kernel_size"), enc["conv_kernel_size"]),
            (e.get("subsampling_factor"), enc["subsampling_factor"]),
            (e.get("subsampling_conv_channels"), enc["subsampling_conv_channels"]),
            (e.get("num_mel_bins"), pre["features"]),
            (e.get("scale_input"), enc.get("xscaling")),
            (e.get("attention_bias"), enc.get("use_bias")),
            (h.get("decoder_hidden_size"), cfg["decoder"]["prednet"]["pred_hidden"]),
            (h.get("num_decoder_layers"), cfg["decoder"]["prednet"]["pred_rnn_layers"]),
            (h.get("vocab_size"), cfg["decoder"]["vocab_size"] + 1),
            (
                list(h.get("durations", [])),
                list(cfg["model_defaults"]["tdt_durations"]),
            ),
        ]
        bad = [
            "%r, the base's %r" % (have, base) for have, base in want if have != base
        ]
        if bad:
            raise ValueError(
                "%s is not its base's network: %s" % (self.path, "; ".join(bad))
            )

    @property
    def state(self):
        """NeMo's state dict, float32: the HF tensors under NeMo's names
        (float16 ones widened, which is exact), the base's buffers."""
        if self._state is None:
            st, self.extra = {}, []
            for name, arr in safetensors(self.path).items():
                n = nemo_name(name)
                if n is None:
                    self.extra.append(name)
                    continue
                st[n] = arr.astype(np.float32) if arr.dtype == np.float16 else arr
            bs = self.base.state
            for n in FROM_BASE:
                st[n] = bs[n]
            missing = sorted(set(bs) - set(st))
            if missing:
                raise ValueError("%s lacks the base's %s" % (self.path, missing[:8]))
            shapes = [n for n in bs if tuple(np.shape(bs[n])) != tuple(np.shape(st[n]))]
            if shapes:
                raise ValueError(
                    "%s: not the base's shapes: %s" % (self.path, shapes[:8])
                )
            unknown = sorted(set(st) - set(bs))
            if unknown:
                raise ValueError(
                    "%s: tensors the base has none of: %s" % (self.path, unknown[:8])
                )
            self._state = st
        return self._state

    def tokenizer_model(self):
        return self.base.tokenizer_model()

    def check_tokenizer(self, pieces):
        """tokenizer.json's vocabulary is the base SentencePiece model's
        pieces, each at its id."""
        p = os.path.join(os.path.dirname(os.path.abspath(self.path)), "tokenizer.json")
        with open(p, encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
        ids = (
            vocab.items()
            if isinstance(vocab, dict)
            else ((v[0], i) for i, v in enumerate(vocab))
        )
        bad = [(t, i) for t, i in ids if i < len(pieces) and pieces[i] != t]
        if bad or len(vocab) < len(pieces):
            raise ValueError(
                "%s is not the base's tokenizer (%d pieces of %d differ)"
                % (p, len(bad), len(pieces))
            )


def open_checkpoint(path, base=None):
    """A .nemo, or a model.safetensors with the .nemo it was made from."""
    if path.endswith(".safetensors"):
        if not base:
            raise ValueError("%s needs --nemo-base, the .nemo it was made from" % path)
        return HF(path, base)
    return nemo.Nemo(path)
