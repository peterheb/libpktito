"""A SentencePiece model file read with the protobuf wire format alone, and
its decoder.

Only the decode direction matters here: the engine turns token ids into
text and never text into ids. That needs each piece's string and type and
three settings, from sentencepiece_model.proto:

    ModelProto        pieces = 1, trainer_spec = 2, normalizer_spec = 3
    SentencePiece     piece = 1, score = 2, type = 3
    TrainerSpec       model_type = 3, vocab_size = 4, byte_fallback = 35,
                      unk_surface = 44
    NormalizerSpec    add_dummy_prefix = 3, remove_extra_whitespaces = 4

decode() is sentencepiece's DecodeIds as its C++ spells it, written out so
that the C detokenizer (core/tokens.c) has a definition to be held to.
"""
from .onnxpb import fields

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
TYPE_NAMES = {NORMAL: "normal", UNKNOWN: "unknown", CONTROL: "control",
              USER_DEFINED: "user", UNUSED: "unused", BYTE: "byte"}
SPACE = "▁"          # the piece's spelling of a space


class Model:
    def __init__(self, data):
        self.pieces, self.types = [], []
        self.model_type, self.byte_fallback = 1, False
        self.unk_surface = " ⁇ "
        self.add_dummy_prefix, self.remove_extra_whitespaces = True, True
        for f, wt, v in fields(data):
            if f == 1 and wt == 2:
                piece, typ = "", NORMAL
                for ff, wwt, vv in fields(v):
                    if ff == 1 and wwt == 2:
                        piece = bytes(vv).decode("utf-8")
                    elif ff == 3 and wwt == 0:
                        typ = vv
                if typ not in TYPE_NAMES:
                    raise ValueError("piece %d has unknown type %d" % (len(self.pieces), typ))
                self.pieces.append(piece)
                self.types.append(typ)
            elif f == 2 and wt == 2:
                for ff, wwt, vv in fields(v):
                    if ff == 3 and wwt == 0:
                        self.model_type = vv
                    elif ff == 35 and wwt == 0:
                        self.byte_fallback = bool(vv)
                    elif ff == 44 and wwt == 2:
                        self.unk_surface = bytes(vv).decode("utf-8")
            elif f == 3 and wt == 2:
                for ff, wwt, vv in fields(v):
                    if ff == 3 and wwt == 0:
                        self.add_dummy_prefix = bool(vv)
                    elif ff == 4 and wwt == 0:
                        self.remove_extra_whitespaces = bool(vv)
        if not self.pieces:
            raise ValueError("no pieces: not a SentencePiece model")

    def decode(self, ids):
        """sentencepiece's Decode (sentencepiece_processor.cc) for a model
        without byte pieces, which the converter refuses: a control piece is
        nothing, the unknown piece is unk_surface, any other piece is its
        text with its space symbols made spaces. While nothing has been
        written and no leading space consumed, a piece's leading space symbol
        is consumed, when the model adds a dummy prefix or removes extra
        whitespace; removing extra whitespace keeps consuming them until text
        appears."""
        out, seen = [], False
        for i in ids:
            typ, piece = self.types[i], self.pieces[i]
            if typ == BYTE:
                raise ValueError("byte pieces are not supported")
            bos_ws = not seen and not out
            seen = False
            if typ == CONTROL:
                continue
            if typ == UNKNOWN:
                s = self.unk_surface
            else:
                if bos_ws and (self.add_dummy_prefix or self.remove_extra_whitespaces) \
                        and piece.startswith(SPACE):
                    piece = piece[1:]
                    seen = not self.remove_extra_whitespaces
                s = piece.replace(SPACE, " ")
            if s:
                out.append(s)
        return "".join(out)
