"""A NeMo checkpoint (.nemo) read with the standard library and numpy.

A .nemo is a tar holding model_config.yaml, model_weights.ckpt (a torch zip
checkpoint, torchzip.py) and the tokenizer's files, which the config names
as "nemo:<member>". Nothing is extracted: the tar's members are read in
place, the checkpoint through a seekable view of its member.
"""
import tarfile

from . import torchzip, yamlite


class Nemo:
    def __init__(self, path):
        self.path = path
        self.tar = tarfile.open(path, "r:*")
        self.members = {m.name.lstrip("./"): m for m in self.tar.getmembers() if m.isfile()}
        self.config = yamlite.loads(self.read("model_config.yaml").decode("utf-8"))
        self._state = None

    def read(self, name):
        name = name[5:] if name.startswith("nemo:") else name
        if name not in self.members:
            raise KeyError("%s has no member %s" % (self.path, name))
        return self.tar.extractfile(self.members[name]).read()

    @property
    def state(self):
        """The state dict, {name: numpy array}, read on first use."""
        if self._state is None:
            self._state = torchzip.load(self.tar.extractfile(self.members["model_weights.ckpt"]))
        return self._state

    def tokenizer_model(self):
        return self.read(self.config["tokenizer"]["model_path"])
