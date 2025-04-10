# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.

"""
Functions for import and export.

This module provides the ability to load vector fields from file and save
vector fields and vector rose histogram data.
"""

import enum
import os
from typing import Any, Dict, Optional, Sequence, Type, Union

import matplotlib.animation
import numpy as np
import pandas as pd

DEFAULT_LOCATION_COLUMNS = (0, 1, 2)
"""Default column numbers for the location coordinates in the order 
``(x, y, z)``."""

DEFAULT_COMPONENT_COLUMNS = (-3, -2, -1)
"""Default column numbers for the vector components in the order 
``(vx, vy, vz)``."""


class VectorFileType(enum.Enum):
    """File types for numeric data.

    Numeric data may be imported and exported in a number of different
    formats. This enumerated type allows the user to specify which file
    type they would like to use to load or store numeric data, such as
    vector lists and binning arrays. The associated strings for each member
    are the file extension **without a dot**.

    Members
    -------
    CSV
        Comma-separated value file, in which the columns will be separated
        by a tab "\\t". File extension: ``*.csv``.
    NPY
        NumPy array, which can easily be loaded into NumPy. File extension:
        ``*.npy``.
    EXCEL
        Microsoft Excel spreadsheet (compatible with Excel 2007 or later).
        File extension: ``*.xlsx``.

    Warnings
    --------
    When constructing a filename using the members of this type, a dot
    ``(.)`` must be added.
    """

    CSV = "csv"
    NPY = "npy"
    EXCEL = "xlsx"


class ImageFileType(enum.Enum):
    """Image File Types.

    File types for images. These include both raster formats (``*.png`` and
    ``*.tiff``) and vector formats (``*.svg`` and ``*.pdf``). The members
    of this enumerated type have as value the string extensions for the
    respective file types **without** the dot.

    Members
    -------
    PNG
        Portable Network Graphics (png) image (raster).
    TIFF
        Tagged Image File Format (tiff) image (raster).
    SVG
        Scalable Vector Graphic (svg) image (vector).
    PDF
        Portable Document Format (pdf) file (vector).

    Warnings
    --------
    When constructing a filename using the members of this type, a dot
    ``(.)`` must be added.
    """

    PNG = "png"
    TIFF = "tiff"
    SVG = "svg"
    PDF = "pdf"


class VideoFileType(enum.Enum):
    """Video File Types.

    File types for videos and animations. The raw values for the members of
    this enumerated type correspond to the respective file extensions
    **without** a period.

    Members
    -------
    MP4
        Moving Picture Experts Group (MPEG) 4 video format.
    GIF
        Graphics Interchange Format animated image (regardless of whether
        you pronounce if G-IF or J-IF).


    Warnings
    --------
    When constructing a filename using the members of this type, a dot
    ``(.)`` must be added.
    """

    MP4 = "mp4"
    GIF = "gif"


def __infer_filetype_from_filename(
    filename: str, file_type_enum: Type[enum.Enum]
) -> Optional[enum.Enum]:
    """Infer a file type from a filename.

    This function tries to infer a file type, of the provided  enumerated
    type ``file_type_enum`` from a provided filename by checking the
    extension. If no valid extension is found, ``None`` is returned.
    Otherwise, the determined file type is returned.

    Parameters
    ----------
    filename
        String containing the filename.
    file_type_enum
        Enumerated type representing the desired file type. This enumerated
        type should have string values representing various file
        extensions. These values **should not** contain a dot.

    Returns
    -------
    file_type_enum or None:
        Member of ``file_type_enum`` if a valid file type is found.
        Otherwise, ``None``.

    See Also
    --------
    ImageFileType: Sample enumerated types to pass in for image files.
    VectorFileType: Sample enumerated types for vector data files.
    """

    # Separate out the file extension
    basename, extension = os.path.splitext(filename)

    # Remove the dot from the extension.
    cleaned_extension = extension.lstrip(".")

    try:
        # Try to get the file type based on the extension.
        file_type = file_type_enum(cleaned_extension)
    except ValueError:
        # Otherwise, no filetype found.
        file_type = None

    return file_type


def __infer_vector_filetype_from_filename(
    filename: str,
) -> Optional[VectorFileType]:
    """Infer a vector field file type from a filename.

    This function tries to infer a :class:`VectorFileType` from a provided
    filename by checking the extension. If no valid extension is found,
    :class:`None` is returned. Otherwise, the determined vector type is
    returned.

    Parameters
    ----------
    filename
        String containing the filename.

    Returns
    -------
    VectorFileType or None:
        Vector file type corresponding to the filename if a valid filetype
        is found. Otherwise, :class:`None`.
    """

    vector_file_type = __infer_filetype_from_filename(
        filename=filename, file_type_enum=VectorFileType
    )

    return vector_file_type


def __infer_video_filetype_from_filename(
    filename: str,
) -> Optional[VideoFileType]:
    """Infer a video file type from a filename.

    This function tries to infer a :class:`VideoFileType` from a provided
    filename by checking the extension. If no valid extension is found,
    :class:`None` is returned. Otherwise, the determined vector type is
    returned.

    Parameters
    ----------
    filename
        String containing the filename.

    Returns
    -------
    VideoFileType or None:
        Video file type corresponding to the filename if a valid filetype
        is found. Otherwise, :class:`None`.
    """

    video_file_type = __infer_filetype_from_filename(
        filename=filename, file_type_enum=VideoFileType
    )

    return video_file_type


def import_vector_field(
    filepath: str,
    default_file_type: VectorFileType = VectorFileType.NPY,
    contains_headers: bool = False,
    sheet: Optional[Union[str, int]] = None,
    location_columns: Optional[Sequence[int]] = DEFAULT_LOCATION_COLUMNS,
    component_columns: Sequence[int] = DEFAULT_COMPONENT_COLUMNS,
    component_axis: int = -1,
    separator: str = "\t",
) -> Optional[np.ndarray]:
    """Import a vector field.

    Load a vector field from a file into a NumPy array. For available
    file formats, see :class:`VectorFileType`. The file type is inferred
    from the filename. If it cannot be inferred, the ``default_file_type``
    is tried. If the vector field is not valid, then :class:`None` is
    returned.

    Parameters
    ----------
    filepath
        File path to the vector field file.
    default_file_type
        File type to attempt if the type cannot be inferred from the
        filename.
    contains_headers
        Indicate whether the file contains headers. This option is only
        considered if the vectors are in a CSV or Excel file.
    sheet
        Name or index of the sheet to consider if the vectors are in an
        Excel file.
    location_columns
        Column indices for the vector *spatial coordinates* in the order
        ``(x, y, z)``. If this is set to :class:`None`, the vectors are
        assumed to be located at the origin. By default, the first three
        columns are assumed to refer to ``(x, y, z)``, respectively.
    component_columns
        Column indices referring to the vector *components* in the order
        ``(vx, vy, vz)``. By default, the last three columns
        ``(-3, -2, -1)`` are assumed to be the ``(vx, vy, vz)``.
    component_axis
        Axis along which the components are defined, in the case of a NumPy
        array which has more than 2 dimensions.
    separator
        Column separator to use if the vector field is a CSV file.

    Returns
    -------
    numpy.ndarray or None
        NumPy array containing the vectors. The array has shape
        ``(n, 3)`` or ``(n, 6)``, depending on whether the locations
        are included. The columns correspond to ``(x,y,z)`` coordinates
        of the location (if available), followed by ``(vx, vy, vz)``
        components. If the filetype cannot be properly inferred,
        a value of ``None`` is returned instead.

    Raises
    ------
    OSError
        The requested file does not exist.
    ValueError
        The requested file cannot be parsed.

    Notes
    -----
    If the vector field file passed contains an array with a dimension > 2,
    then the components are assumed to be along the axis passed in the
    argument `component_axis` and the array will be flattened to 2D.
    """

    # First, infer the file type from the filename
    filetype = __infer_vector_filetype_from_filename(filepath)

    # If inference fails, try the default file type.
    if filetype is None:
        filetype = default_file_type

    if filetype is VectorFileType.NPY:
        vector_field: np.ndarray = np.load(filepath)

        # Check the dimension of the vector field array
        if vector_field.ndim > 2:
            # Get the vector dimension
            d = vector_field.shape[component_axis]

            # Flatten the array
            vector_field = np.moveaxis(vector_field, component_axis, -1).reshape(
                -1, d
            )

        # Remove any rows containing NaN
        vector_field = vector_field[~np.any(np.isnan(vector_field), axis=1)]

    # Use Pandas in the other cases
    else:
        header_row: Optional[int] = 0 if contains_headers else None
        # Reading function depends on whether CSV or Excel
        if filetype is VectorFileType.CSV:
            vector_field_dataframe = pd.read_csv(
                filepath, header=header_row, sep=separator
            )
        elif filetype is VectorFileType.EXCEL:
            vector_field_dataframe = pd.read_excel(
                filepath, sheet_name=sheet, header=header_row
            )
        else:
            return None

        # Remove NaN values
        vector_field = vector_field_dataframe.dropna().to_numpy()

    n, d = vector_field.shape

    # Now, for the column parsing
    if location_columns is None or d < 6:
        # No location, only consider the components
        clean_vector_field = vector_field[:, component_columns]
    else:
        # Consider both the location and the components.
        column_indices = list(location_columns) + list(component_columns)

        # Squeeze is necessary to not break type safety.
        clean_vector_field = vector_field[:, column_indices]

    # Convert the vector field to have high-precision floating point type
    clean_vector_field = clean_vector_field.astype(float)

    return clean_vector_field


def export_mpl_animation(
    animation: matplotlib.animation.Animation,
    filename: str,
    file_type: VideoFileType = VideoFileType.MP4,
    dpi: Optional[int] = 150,
    fps: Optional[int] = None,
    **export_kwargs: Dict[str, Any],
):
    """Export a Matplotlib animation.

    Export a provided Matplotlib animation as a video.

    Parameters
    ----------
    animation
        The :class:`matplotlib.animation.Animation` animation to export.
    filename
        The destination for the video export. This filename will be used to
        infer the export file type.
    file_type
        The default filetype to consider if unable to resolve the file type
        from the file name.
    dpi
        Resolution of the exported video in dots-per-inch (DPI).
    fps
        Desired frame rate in the exported video.
    export_kwargs
        Additional keyword arguments
        to :meth:`matplotlib.animation.Animation.save`.

    See Also
    --------
    matplotlib.animation.Animation.save:
        The function called to perform the actual export.

    """

    # Infer the file type from the filename
    inferred_type = __infer_video_filetype_from_filename(filename)

    if inferred_type is None:
        filename = f"{filename}.{file_type.value}"

    if fps == 0:
        fps = None

    animation.save(filename=filename, fps=fps, dpi=dpi, **export_kwargs)
