"""
Dear Sequence Dataframe,

I would like you to:
- hold sequential data for me
- ON DIMENSIONS:
-- the raw data can either have 1 dimension: N
-- or 2 dimensions: (N, timestep) where timestep might not be constant!
-- learning and testing should have 3 dimensions: (n, timestep, embeddim)
!! NOPE !! the returned table should be dimshuffled so: (timestep, n, embeddim)
This must be so because batch shuffling should be done on n, but RNNs require
the timestep to be the first dim to be able to "batch" them efficiently.

- timestep (aka the length of a sequence element) can vary!
- embeddim should be constant and should only be returned by <neurons_required>
- implement embedding on the independent variables (X)!
? maybe implement transformations on embedded X?

- a parser is also needed for text data and for DNA sequence data.

"""

import unittest


class TestSeq(unittest.TestCase):
    pass
