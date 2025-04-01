---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Loading Vectors into VectoRose

The previous section introduced axial and vectorial data. VectoRose is a
Python package that can be used to visualise and analyse these data. But,
before the data can be examined, the vectors must be loaded into VectoRose.

This page describes how vectors must be formatted and how to import them
into VectoRose. Unlike images, which have very well-defined standards,
vectorial data have yet to be widely standardised. We have tried to define
simple, intuitive formats for representing vectorial data.

## Data Formats and Layout

VectoRose accepts axial and vectorial data in three formats:
1. Binary NumPy files (`*.npy`)
2. Comma-separated value files (`*.csv` or `*.txt`)
3. Excel spreadsheets (`*.xlsx`)

The data must be arranged so that each **row** represents a single vector,
and the columns represent the vector components. This diagram illustrates
how the file should be organised:

![Data table representation showing vectors as rows and
components as columns](assets/data_format/VectorFormatting.png)

If the provided vectors contain spatial information, the first three
columns are assumed to represent the vector positions in space, while the
last three columns are assumed to represent the vector components. While
these settings are the default, they can be easily customised to
accommodate files produced by other software tools.

```{attention}
While it is possible to configure these options when loading a file, once a
collection of vectors is open in VectoRose, these are the conventions that
are followed.
```

### NumPy Arrays

NumPy binary array files (see
[here](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
for more detail) are a flexible, efficient way of storing multidimensional
arrays. The important trade-off is that the file format is *binary*, and so
you can't open these files in a text editor.

#### Higher-dimensional Arrays

Unlike spreadsheets and text-based files, the information stored in NumPy
arrays are not restricted to two dimensions. NumPy files can easily store
arrays of any dimension. An example of a higher dimensional array is a
*vector field*. Since a vector is defined at each position in 3D space, it
may be more intuitive to store these data in a 4D array, where three of the
dimensions represent the spatial location and the fourth is used to
distinguish the vector components.

```{warning}
Currently, only `*.npy` files can be imported. To import an array stored in
a compressed `*.npz` file, extract the constituent arrays and load the
specific `*.npy` file extracted.
```

### CSV Files

CSV files are a **plain-text** format which represent data as a 2D table.
Each line in the file represents a table row. Within a row, columns are
separated by a specific character, such as a comma (`,`), a tab (`\t`), a
space (` `) or a semicolon (`;`). As these files are text-based, they are
relatively lightweight and can easily be opened with a wide variety of
editing software[^text-editors].

### Excel Spreadsheets

Excel spreadsheets are a more sophisticated XML-based format for storing
multiple 2D tables as sheets containing rows and columns. Similar to the
other formats described, rows represent different vectors and columns
represent different vector components.

```{warning}
VectoRose can only open the newer ``*.xlsx`` files. The older ``*.xls``
spreadsheets may not be supported.
```

### Data Export Software Plugins

We have created a plugin for the [*Dragonfly 3D World*](https://dragonfly.comet.tech/)
image analysis software developed by Comet Technologies Canada, Inc. to
export vector fields in these formats. This plugin can be downloaded from
our online repository (**COMING SOON**).

```{attention}
This plugin is not developed or supported by Comet Technologies Canada,
Inc. We provide this plugin completely independently and we are solely
responsible for its development. Except as required by law, this extension
is provided with NO WARRANTY.
```

```{note}
Have another software tool that generates vectors? Feel free to write a
plugin and share the link with us so that we can share it with the
community.
```

## Importing Vectors into VectoRose

Once we have vectors in one of the file formats described above, we can
load these vectors into Python using VectoRose.

Before trying to load your vectors, you **must** import the `vectorose`
package into the Python interpreter:

```python
import vectorose as vr
```

```{tip}
To save some time writing code, we recommend using `vr` as a shorthand for
`vectorose`.
```

Vectors are imported using the function
{func}`vectorose.io.import_vector_field`. For example, if your vectors are
in a NumPy array file called `random_vectors.npy`, we can load the vectors
by writing:

```{code-cell} ipython3
import vectorose as vr

vectors = vr.io.import_vector_field("random_vectors.npy")

vectors
```

We can now see that we have an array of vectors available to process and
analyse.

There are a number of parameters that can control how the vectors are
loaded. The most important parameters are:

`component_columns`
: Indicate the columns containing the `x,y,z` vector 
  components. By default, the last three columns are considered.

`location_columns`
: Indicate the columns containing the `x,y,z`
  positions of the vectors in space, in the case of a vector field. If this 
  is set to `None`, then the location coordinates are ignored. By default,
  the first three columns are considered.

`separator`
: When reading vectors from a **CSV file**, indicate what
  character is used to separate the columns.

`contains_headers`
: Indicate whether the first row of the file contains
  column headers, which will be discarded.

`sheet`
: When reading vectors from an **Excel file**, indicate the name
  or position of the sheet to read.

Regardless of the file type, this function creates a 2D NumPy array with
`n` rows, corresponding to the number of vectors, and either 3 or 6
columns, depending on whether the location coordinates are read.

## Pre-processing Vectors

Once the vectors are read, there are a number of important pre-processing
steps that can be performed:

{func}`vectorose.util.remove_zero_vectors`
: Remove all vectors with a magnitude of zero from the list.

{func}`vectorose.util.convert_vectors_to_axes`
: Flip all vectors having a negative `z`-component to ensure that all
  orientations are contained within the upper unit hemisphere.

{func}`vectorose.util.create_symmetric_vectors_from_axes`
: When analysing axial data, generate a pair of antiparallel vectors for 
  each vector in the list.

{func}`vectorose.util.normalise_vectors`
: Return a set of unit vectors having the same orientations/directions as
  the loaded data.

## Example

We have a collection of vectors in [`random_vectors.csv`](./random_vectors.csv).
Take a look at this file... the columns are separated by commas and the
first row is a header, and there are no spatial coordinates present.

Let's load these vectors, remove any zero-vectors and convert these vectors
into an axial representation. Here's how we can perform this task:

```{code-cell} ipython3
import vectorose as vr

# Load the vectors from the CSV file
vectors = vr.io.import_vector_field(
    "random_vectors.csv", contains_headers=True, location_columns=None, separator=","
)

print(f"We have loaded {vectors.shape[0]} vectors from the file.")

# Remove zero-magnitude vectors
vectors = vr.util.remove_zero_vectors(vectors)
print(f"We have {vectors.shape[0]} non-zero vectors.")

# Convert to axial data
vectors = vr.util.convert_vectors_to_axes(vectors)

vectors
```

We can now see that we've loaded the vectors, and we've managed to prune
quite a few zero-vectors that we had in our dataset.

```{seealso}
For more details about importing vector fields, check out the
documentation on {mod}`vectorose.io` and for more on pre-processing,
consult the page on {mod}`vectorose.util`.
```

But, loading vectors is just the beginning! Now that we know how to load
and pre-process vectors, we can begin with data visualisation.

[^text-editors]: And we'll politely sit out the fight over which text
editor that would be...
