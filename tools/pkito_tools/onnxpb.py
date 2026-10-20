"""Just enough of the protobuf wire format to read an ONNX file's tensors.

vadito's reader (tools/vadito_tools/onnxpb.py), unchanged: pkito reads
Silero VAD's weights out of silero_vad.onnx with it, and the SentencePiece
model with its fields().

An ONNX file is one protobuf message, and the onnx package is protobuf's
runtime plus a compiled schema. Reading the weights back needs neither. A
message is a run of records, each a varint key (field number << 3 | wire
type) and a value, and the path from the file to a tensor crosses five
messages and a dozen field numbers between them, all from onnx/onnx.proto:

    ModelProto      graph = 7, functions = 25
    GraphProto      node = 1, initializer = 5
    FunctionProto   node = 7
    NodeProto       output = 2, op_type = 4, attribute = 5
    AttributeProto  name = 1, t = 5, g = 6, tensors = 10, graphs = 11
    TensorProto     dims = 1, data_type = 2, float_data = 4, int32_data = 5,
                    name = 8, raw_data = 9, double_data = 10, data_location = 14

Silero's own exports keep their weights in three different places: as
Constant nodes inside an If node's two branches (silero_vad.onnx, written by
spox), as graph initializers (the op15 and op18 exports, written by
torch.onnx), and as Constants in the main graph (the OpenVINO export).
tensors() walks all of them, subgraphs and local functions included, and
yields every floating-point tensor under the name the graph knows it by.
"""
import numpy as np

# TensorProto.DataType values (onnx.proto)
FLOAT, FLOAT16, DOUBLE = 1, 10, 11
_EXTERNAL = 1                       # TensorProto.DataLocation.EXTERNAL

# AttributeProto field numbers for the values that can hold a tensor
_ATTR_T, _ATTR_G, _ATTR_TENSORS, _ATTR_GRAPHS = 5, 6, 10, 11


class FormatError(ValueError):
    pass


def _varint(b, i):
    x = shift = 0
    while True:
        if i >= len(b):
            raise FormatError('varint runs past the end of its message')
        c = b[i]
        i += 1
        x |= (c & 0x7F) << shift
        if c < 0x80:
            return x, i
        shift += 7
        if shift > 63:
            raise FormatError('varint longer than ten bytes')


def fields(b):
    """(field number, wire type, value) for each record of one message: an
    int for the varint and fixed-width types, a memoryview for a
    length-delimited one."""
    b = memoryview(b)
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1 or wt == 5:
            n = 8 if wt == 1 else 4
            if len(b) - i < n:
                raise FormatError('fixed field runs past the end of its message')
            v = int.from_bytes(b[i:i + n], 'little')
            i += n
        elif wt == 2:
            n, i = _varint(b, i)
            if n > len(b) - i:
                raise FormatError('length-delimited field runs past the end of its message')
            v = b[i:i + n]
            i += n
        else:
            # 3 and 4 are proto2 groups, which nothing in ONNX uses
            raise FormatError('wire type %d at field %d' % (wt, f))
        yield f, wt, v


def _ints(wt, v):
    """A repeated integer field arrives packed (one length-delimited record)
    or unpacked (one varint record per element); both are legal."""
    if wt == 0:
        return [v]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(x)
    return out


def tensor(b):
    """(name, array) for one TensorProto, or (name, None) if it is not a
    floating-point tensor."""
    name, dims, dtype = '', [], 0
    raw = None
    floats, doubles, int32s = [], [], []
    for f, wt, v in fields(b):
        if f == 1:
            dims += _ints(wt, v)
        elif f == 2:
            dtype = v
        elif f == 4:
            floats.append(np.frombuffer(v, '<f4') if wt == 2 else
                          np.array([v], '<u4').view('<f4'))
        elif f == 5:
            int32s += _ints(wt, v)
        elif f == 8:
            name = bytes(v).decode('utf-8')
        elif f == 9:
            raw = bytes(v)
        elif f == 10:
            doubles.append(np.frombuffer(v, '<f8') if wt == 2 else
                           np.array([v], '<u8').view('<f8'))
        elif f == 14 and v == _EXTERNAL:
            raise FormatError('tensor %r keeps its data in an external file' % name)
    np_type = {FLOAT: '<f4', FLOAT16: '<f2', DOUBLE: '<f8'}.get(dtype)
    if np_type is None:
        return name, None
    count = int(np.prod(dims)) if dims else 1
    if raw is not None:
        a = np.frombuffer(raw, np_type)
    elif dtype == FLOAT:
        a = np.concatenate(floats) if floats else np.zeros(0, '<f4')
    elif dtype == DOUBLE:
        a = np.concatenate(doubles) if doubles else np.zeros(0, '<f8')
    else:
        # fp16 without raw_data: one int32_data element per value, low half
        a = np.array(int32s, '<u4').astype('<u2').view('<f2')
    if a.size != count:
        raise FormatError('tensor %r holds %d values for shape %s' % (name, a.size, dims))
    return name, a.reshape(dims)


def _graph(b, out):
    for f, wt, v in fields(b):
        if f == 1 and wt == 2:
            _node(v, out)
        elif f == 5 and wt == 2:
            name, a = tensor(v)
            if a is not None:
                out.append((name, a))


def _node(b, out):
    outputs, op, attrs = [], '', []
    for f, wt, v in fields(b):
        if f == 2 and wt == 2:
            outputs.append(bytes(v).decode('utf-8'))
        elif f == 4 and wt == 2:
            op = bytes(v).decode('utf-8')
        elif f == 5 and wt == 2:
            attrs.append(v)
    for a in attrs:
        attr_name = ''
        for f, wt, v in fields(a):
            if f == 1 and wt == 2:
                attr_name = bytes(v).decode('utf-8')
        for f, wt, v in fields(a):
            if wt != 2:
                continue
            if f in (_ATTR_T, _ATTR_TENSORS):
                name, t = tensor(v)
                if t is None:
                    continue
                # a Constant's tensor is known to the graph by the node's
                # output, whatever (often nothing) its own name field says
                if op == 'Constant' and attr_name == 'value' and outputs:
                    name = outputs[0]
                out.append((name, t))
            elif f in (_ATTR_G, _ATTR_GRAPHS):
                _graph(v, out)


def tensors(path):
    """Every floating-point tensor in an ONNX model file, as (name, array)."""
    with open(path, 'rb') as fh:
        b = fh.read()
    out = []
    for f, wt, v in fields(b):
        if f == 7 and wt == 2:
            _graph(v, out)
        elif f == 25 and wt == 2:
            for ff, wwt, vv in fields(v):
                if ff == 7 and wwt == 2:
                    _node(vv, out)
    return out
