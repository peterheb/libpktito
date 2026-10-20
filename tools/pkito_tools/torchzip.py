"""A PyTorch checkpoint (torch.save's zip format) read with numpy alone.

torch.save writes a zip: `<prefix>/data.pkl` is a pickle of the state dict
in which every tensor is a call to torch._utils._rebuild_tensor_v2 on a
"persistent id" naming a storage, and `<prefix>/data/<key>` is that
storage's raw little-endian bytes. Unpickling it the ordinary way imports
torch. This unpickler instead answers the handful of globals a state dict
uses with stand-ins that build numpy arrays, and refuses every other one,
so a checkpoint cannot run code through it: find_class is the whole of
what pickle can call.
"""
import collections
import pickle
import zipfile

import numpy as np

# torch's storage classes, by the name the pickle uses, and their dtype
_STORAGES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "BFloat16Storage": "bf16",
}


class _Storage:
    def __init__(self, dtype):
        self.dtype = dtype


class _Ref:
    """A storage named by a persistent id, read when a tensor needs it."""
    def __init__(self, zf, prefix, key, dtype):
        self.zf, self.name, self.dtype = zf, "%s/data/%s" % (prefix, key), dtype
        self.arr = None

    def array(self):
        if self.arr is None:
            raw = self.zf.read(self.name)
            if self.dtype == "bf16":
                u = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
                self.arr = u.view(np.float32)
            else:
                self.arr = np.frombuffer(raw, np.dtype(self.dtype).newbyteorder("<"))
        return self.arr


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, meta=None):
    a = storage.array()
    size, stride = tuple(int(s) for s in size), tuple(int(s) for s in stride)
    n = 1
    for s in size:
        n *= s
    if n == 0:
        return np.zeros(size, a.dtype)
    last = offset + sum((s - 1) * st for s, st in zip(size, stride))
    if offset < 0 or last >= a.size or any(st < 0 for st in stride):
        raise ValueError("a tensor reaches outside its storage")
    view = np.lib.stride_tricks.as_strided(
        a[offset:], shape=size, strides=tuple(st * a.itemsize for st in stride), writeable=False)
    return np.array(view)


def _rebuild_parameter(data, requires_grad=False, hooks=None):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.refs = zf, prefix, {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGES:
            return _Storage(_STORAGES[name])
        raise pickle.UnpicklingError("%s.%s is not something a state dict holds" % (module, name))

    def persistent_load(self, pid):
        # ('storage', storage type, key, location, numel)
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"
                and isinstance(pid[1], _Storage)):
            raise pickle.UnpicklingError("unexpected persistent id %r" % (pid,))
        key = str(pid[2])
        if key not in self.refs:
            self.refs[key] = _Ref(self.zf, self.prefix, key, pid[1].dtype)
        return self.refs[key]


def load(fileobj):
    """The state dict in a torch zip checkpoint, as {name: numpy array}.
    `fileobj` is a path or a seekable binary file."""
    zf = zipfile.ZipFile(fileobj)
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl") and n.count("/") == 1]
    if len(pkl) != 1:
        raise ValueError("not a torch zip checkpoint: no single <prefix>/data.pkl")
    prefix = pkl[0].split("/")[0]
    with zf.open(pkl[0]) as f:
        sd = _Unpickler(f, zf, prefix).load()
    if not isinstance(sd, dict):
        raise ValueError("the checkpoint is not a state dict")
    return collections.OrderedDict((k, v) for k, v in sd.items())
