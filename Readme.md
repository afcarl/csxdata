#Dataframes for holding and interacting with data.

Mainly used to interface with neural networking libraries and Scikit-learn.

Four dataframes are currently available in the *frames* module.
Besides dataframes, several wrappers exist to read, preprocess and inspect.

##Frames
At instantiation time, there is a lot of implicit guesswork going on about
the source supplied by the user. The optimal parser is guessed based on the
format of the input.

The number of samples isolated for cross-validation can be set by setting the
**crossval** property to the ratio (*float*) or the number itself (*integer*)
The preprocessing can be set by assigning a tuple (transformation name: string,
number of features to get: int) to the property **transformation** or by the
method **set_transformation(name, param)**.
Currently standardization, PCA, ICA, LDA, PLS via sklearn and autoencoding via
keras are supported.

The various transformed datasets can be accessed via the **table(data, m, shuff)**
method (dataset can be specified by setting the *data* parameter to "testing" or "learning").
The parameters m and shuff set the number of samples to return and whether or not to
shuffle them. The returned data is held in RAM.

The method **batchgen(bsize, data, infinite)** returns a generator, which yields
samples with a batch size of *bsize* from the dataset *data* and whether to loop around
after reaching the last sample (*infinite*)

The properties **neurons_required**, **dimensionality** and **N** are used to interface with
the machine learning and neural networking libraries **Sklearn**, **Keras** and **Brainforge**

###CData
Frame for holding categorical data.

Implicitly converts the dependent variables (**Y**) to one-hot vector representations.
Embedding **Y** into k-dimensional space can be done by setting the **embedding** property
to the desired number of dimensions (k) 

###RData
Frame for holding regression data.

Can be used to do regression tasks on Y. Y is allowed to be multi-dimensional.

###Sequence
Frame for holding sequential data.

Not very general implementation. Tested only on text data with Recurrent Neural Networks.
The constructor parameters **n_gram** and **timestep** are used to reorder and chop up the
sequence to form a 3D t, N, d representation of it where t is the timestep's length,
N is the total number of samples (or the current batch size) and d is the embedding dimension
(defaults to one-hot representation.)

###MassiveSequence
Frame for holding sequential data.

Even more incomplete than Sequence, this frame is also used to work with text data, but instead of
holding everything in RAM, it uses generators to read and transform the data on the fly.

##Data utilities

###csxdata.stats
This package can be used to inspect the data (**stats.inspections**) and to assess normality
(**stats.normality**). These serve as convenience wrappers to **SciPy** routines.

###csxdata.utilities

**utilities.parsers** implents file and data parsing functions to convert data to an **X, Y, header**
format from pickled learning tables, text data, csv files or in-memory arrays.

**utilities.misc** contains some routines written in pure Python. Rarely used, they are written
for **brainforge.evolution** when used with PyPy.
``
In **utilities.vectorops** a lot of convencience functions are defined for NumPy. The module
only uses NumPy as dependency.

**utilities.highlevel** include wrappers to more high-level modules like **Matplotlib**, **Keras**,
**sklearn**, **Pillow** and **Theano**.

###csxdata.features
**features.embedding** implements routines which are used by frames to embed dependent variables (**Y**)

**features.transformation** defines wrappers for **Scikit-Learn's** dimensionality reduction and
whitening transformations (standardization, PCA, ICA, LDA, PLS, Autoencoding)