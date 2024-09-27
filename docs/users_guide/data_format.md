# Data Formatting

This page describes how vectors must be formatted in order to import them
into VectoRose. Unlike images, which have very well-defined standards,
vectorial data have yet to be widely standardised. We have tried to define
simple, intuitive formats for representing vectorial data.

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
last three columns are assumed to represent the vector components. If this
assumption is not correct, the **Advanced Open** dialog can be used to
reassign the component columns.

```{eval-rst}
.. seealso::

    For more information on the vector loading process and how it can be
    customised, see the documentation for
    :func:`vectorose.vf_io.import_vector_field`.
```
## NumPy Arrays

NumPy binary array files (see
[here](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
for more detail) are a flexible, efficient way of storing multidimensional
arrays. Unlike spreadsheets and text-based files, the information stored
need not be restricted to two dimensions. NumPy files can thus easily store
arrays of any dimension.

An example of a higher dimensional array is a *vector field*. Since a
vector is defined at each position in 3D space, it may be more intuitive to
store these data in a 4D array, where three of the dimensions represent the
spatial location and the fourth is used to distinguish the vector
components. VectoRose can load such arrays as well. The user must simply
specify which axis represents the vector components.

```{eval-rst}
.. warning::
    Currently, only simple individual ``*.npy`` files are supported. To
    import an array stored in a compressed ``*.npz`` file, extract the
    constituent arrays and load the specific ``*.npy`` file extracted.
```

## CSV Files

CSV files are a **plain-text** format which represent data as a 2D table.
Each line in the file represents a table row. Within a row, columns are
separated by a specific character, such as a comma (`,`), a tab (`\t`), a
space (` `) or a semicolon (`;`). 

```{eval-rst}
.. warning::

    By default, VectoRose assumes that columns are separated by tabs
    (``\\t``). This behaviour can be customised in the **Advanced Open**
    dialog.

```

## Excel Spreadsheets

Excel spreadsheets are a more sophisticated XML-based format for storing
multiple 2D tables as sheets containing rows and columns. Similar to the
other formats described, rows represent different vectors and columns
represent different vector components.

```{eval-rst}

.. warning::
    VectoRose can only open the newer ``*.xlsx`` files. The older ``*.xls``
    spreadsheets may not be supported.

    By default, VectoRose loads the **first sheet** in a spreadsheet file.
    To load a different sheet, use the **Advanced Open** dialog.

```

## Software Plugins

We have created a plugin for the *Dragonfly 3D World* image analysis
software developed by Comet Technologies Canada, Inc. to export vector
fields in these formats. This plugin can be downloaded from our [online
repository](https://github.com/bzrudski/).

```{eval-rst}
.. attention::
    This plugin is not developed or supported by Comet Technologies Canada,
    Inc. They have no role in its development and have not endorsed it. We
    provide this plugin completely independently and we are solely
    responsible for its development.
```

Have another software tool that generates vectors? Feel free to write a
plugin and share the link with us so that we can share it with the
community.
