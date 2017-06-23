YAY = 1.0
NAY = 0.0
UNKNOWN = "<UNK>"


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
        self.mainlog = self.logsroot + "csxdata.log"
        self.txtroot = self.rawroot + "txt/"
        self.sequences = self.rawroot + "seq/"
        self.etalon = self.dataroot + "etalon/"
        self.picsroot = self.rawroot + "pics/"
        self.ntabpics = self.picsroot + "ntab/"
        self.gisroot = self.dataroot + "gis/"

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
                      "mainlog": self.mainlog,
                      "logstring": self.mainlog,
                      "txt": self.txtroot,
                      "text": self.txtroot,
                      "sequence": self.sequences,
                      "sequences": self.sequences,
                      "seq": self.sequences,
                      "etalon": self.etalon,
                      "pics": self.picsroot,
                      "ntabpics": self.ntabpics,
                      "gis": self.gisroot}

    def check_roots(self):
        for name, root in self._dict.items():
            self.root_exists(name, verbose=1)

    def root_exists(self, name, verbose=0):
        from os.path import exists
        if name not in self._dict:
            raise IndexError("Supplied root is not in database!")
        aye = exists(self._dict[name])
        if verbose:
            if aye:
                print("OK: [{}] exists!".format(name))
            else:
                print("!!: [{}] doesn't exist!".format(name))
        return aye

    def create_roots(self, verbose=1):
        from os import mkdir

        def mkroot(path, vbs):
            if vbs:
                print("Creating {}".format(self.dataroot))
            mkdir(path)

        if not self.root_exists(self.dataroot, verbose=verbose):
            mkroot(self.dataroot, verbose)
        for name, root in self._dict.items():
            if root == self.dataroot:
                continue
            if not self.root_exists(name, verbose):
                mkroot(root, verbose)

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
        logfl.write("{}: {}\n".format(dt.now().strftime("%Y.%m.%d %H:%M:%S"), chain))
        logfl.close()
