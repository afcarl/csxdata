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
        self.cacheroot = self.dataroot + ".csxcache/"
        self.logsroot = self.dataroot + ".csxlogs/"
        self.etalon = self.dataroot + "etalon/"
        self.mainlog = self.logsroot + ".csxdata.log"

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
                      "brain": self.brainsroot,
                      "cache": self.cacheroot,
                      "logs": self.logsroot,
                      "etalon": self.etalon,
                      "mainlog": self.mainlog,
                      "log": self.mainlog}

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        if item not in self._dict:
            raise IndexError("Requested path not in database!")

        path = self._dict[item]
        return path

    def __call__(self, item):
        return self.__getitem__(item)


roots = _Roots()


def log(chain):
    from datetime import datetime as dt
    with open(roots["mainlog"], "a") as logfl:
        logfl.write("{}: {}".format(dt.now().strftime("%Y.%m.%d %H:%M:%S"), chain))
        logfl.close()
