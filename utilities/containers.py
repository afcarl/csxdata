class Pipeline:
    """Class for holding data as it was put in a two-ended pipe."""
    def __init__(self, length):
        self.length = length
        self._internals = []

    def add(self, x):
        return self.add_top(x)

    def add_top(self, x):
        self._internals = [x] + self._internals
        if self.elements > self.length:
            return self._internals.pop()

    def add_bottom(self, x):
        self._internals += [x]
        if self.elements > self.length:
            self._internals = self._internals[1:]
            return self._internals[0]

    @property
    def top(self):
        return self._internals[0]

    @property
    def bottom(self):
        return self._internals[-1]

    @property
    def elements(self):
        return len(self._internals)

    @property
    def free(self):
        return self.length - self.elements

    def __len__(self):
        return self.length

    def __str__(self):
        return "Pipeline instance with length {}".format(self.length)

    def __repr__(self):
        return "Pipeline({})".format(self.length)


# TODO: modify add so it can receive an arbitrary number of inputs!
