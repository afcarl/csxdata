YAY = 1.0

NAY = 0.0

UNKNOWN = "<UNK>"

floatX = "float32"


class _Roots:
    def __init__(self):
        import sys
        win = sys.platform == "win32"
        self.dataroot = "D:/Data/" if win else "/data/Prog/data/"
        self.miscroot = self.dataroot + "misc/"
        self.rawroot = self.dataroot + "raw/"
        self.ltroot = self.dataroot + "lts/"
        self.csvroot = self.dataroot + "csvs/"
        self.nirroot = self.rawroot + "nir/"
        self.tmproot = "E:/tmp/" if win else "/run/media/csa/ramdisk/"
        self.hippocrates = self.rawroot + "Project_Hippocrates/"
        self.brainsroot = self.dataroot + "brains/"

        self._dict = {"data": self.dataroot,
                      "raw": self.rawroot,
                      "lt": self.ltroot,
                      "lts": self.ltroot,
                      "csv": self.csvroot,
                      "csvs": self.csvroot,
                      "nir": self.nirroot,
                      "tmp": self.tmproot,
                      "temp": self.tmproot,
                      "misc": self.miscroot,
                      "hippocrates": self.hippocrates,
                      "hippo": self.hippocrates,
                      "brains": self.brainsroot,
                      "brain": self.brainsroot}

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        if item not in self._dict:
            raise IndexError("Requested path not in database!")
        return self._dict[item]

    def __call__(self, item):
        return self.__getitem__(item)

roots = _Roots()
