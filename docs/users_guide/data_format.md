# Loading Vectors into VectoRose

The previous section introduced axial and vectorial data. VectoRose is a
Python package that can analyse these data. But, before the data can be
examined, the vectors must be opened and loaded into VectoRose.

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

### NumPy Arrays

NumPy binary array files (see
[here](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
for more detail) are a flexible, efficient way of storing multidimensional
arrays. The important trade-off is that the file format is *binary*, and so
you can't open these files in a text editor.

#### Higher-dimensional Arrays

Unlike spreadsheets and text-based files, the information stored
need not be restricted to two dimensions. NumPy files can thus easily store
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
space (` `) or a semicolon (`;`).

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

We have created a plugin for the *Dragonfly 3D World* image analysis
software developed by Comet Technologies Canada, Inc. to export vector
fields in these formats. This plugin can be downloaded from our [online
repository](https://github.com/bzrudski/).

```{attention}
This plugin is not developed or supported by Comet Technologies Canada,
Inc. They have no role in its development and have not endorsed it. We
provide this plugin completely independently and we are solely
responsible for its development. Except as required by law, this extension
is provided with NO WARRANTY.
```

Have another software tool that generates vectors? Feel free to write a
plugin and share the link with us so that we can share it with the
community.

## Importing Vectors into VectoRose

Once we have vectors in one of the file formats described above, we can
load these vectors into Python using VectoRose.

Before trying to load your vectors, you **must** import the `vectorose`
package into the Python interpreter:

```python
import vectorose as vr
```

```{note}
To save some time writing code, we recommend using `vr` as a shorthand for
`vectorose`.
```

Vectors are imported using the function
{func}`vectorose.io.import_vector_field`. For example, if your vectors are
in a NumPy array file called `my_vectors.npy`, we can load the vectors by
writing:
```python
vectors = vr.io.import_vector_field("my_vectors.npy")
```

There are a number of parameters that can control how the vectors are
loaded. The most important parameters are:

* `component_columns` - Indicate the columns containing the `x,y,z` vector 
  components. By default, the last three columns are considered.
* `location_columns` - Indicate the columns containing the `x,y,z`
  positions of the vectors in space, in the case of a vector field. If this 
  is set to `None`, then the location coordinates are ignored. By default,
  the first three columns are considered.
* `separator` - When reading vectors from a **CSV file**, indicate what
  character is used to separate the columns.
* `contains_headers` - Indicate whether the first row of the file contains
  column headers, which will be discarded.
* `sheet` - When reading vectors from an **Excel file**, indicate the name
  or position of the sheet to read.

Regardless of the file type, this function creates a 2D NumPy array with
`n` rows, corresponding to the number of vectors, and either 3 or 6
columns, depending on whether the location coordinates are read.

## Pre-processing Vectors

Once the vectors are read, there are a number of important pre-processing
steps that can be performed:
* {func}`vectorose.util.remove_zero_vectors` - Remove all vectors with a
  magnitude of zero from the list.
* {func}`vectorose.util.convert_vectors_to_axes` - Flip all vectors having
  a negative `z`-component to ensure that all orientations are contained
  within the upper unit hemisphere.
* {func}`vectorose.util.create_symmetric_vectors_from_axes` - Using axial
  data, generate a pair of antiparallel vectors for each vector in the
  list.
* {func}`vectorose.util.normalise_vectors` - Return a set of unit vectors
  having the same orientation as the loaded data.

## Example

Let's say we load a collection of vectors from a CSV file called
`my_vectors.csv`. In this file, the columns are separated by a comma and
the first row is a header. We would like to load these vectors without any
location coordinates, remove any zero-vectors and represent them as axial
vectors. Here's how we can perform this task:
```python
import vectorose as vr

# Load the vectors from the CSV file
vectors = vr.io.import_vector_field(
    "my_vectors.csv", contains_headers=True, location_columns=None, separator=","
)

# Remove zero-magnitude vectors
vectors = vr.util.remove_zero_vectors(vectors)

# Convert to axial data
vectors = vr.util.convert_vectors_to_axes(vectors)
```

For a more complete demonstration, make sure to check out our worked
example. For more details about importing vector fields, check out the
documentation on {mod}`vectorose.io` and for more on pre-processing,
consult the page on {mod}`vectorose.util`.

Now that we know how to load and pre-process vectors, we can begin with
data visualisation.
