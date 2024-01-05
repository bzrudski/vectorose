"""
Anisotropy Rose - I/O Operations

Joseph Deering, Benjamin Rudski
2023

This module provides the ability to load vector fields from file and
save vector fields and anisotropy histograms.

"""

import enum
import os
from typing import Optional, Type, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_LOCATION_COLUMNS = (0, 1, 2)
DEFAULT_COMPONENT_COLUMNS = (-3, -2, -1)


class VectorFileType(enum.Enum):
    """
    Export types for numeric data.

    Numeric data may be exported in a number of different formats. This
    enumerated type allows the user to specify which file type they
    would like to use to export numeric data, such as vector lists and
    binning arrays. The associated strings for each time are the file
    extension **without a dot**.

    Attributes:
        * CSV: Comma-separated value file, in which the columns will be
            separated by a tab "\\t". File extension: ``*.csv``.
        * NPY: NumPy array, which can easily be loaded into NumPy. File
            extension: ``*.npy``.
        * EXCEL: Microsoft Excel spreadsheet (compatible with Excel 2007
            or later). File extension: ``*.xlsx``.
    """

    CSV = "csv"
    NPY = "npy"
    EXCEL = "xlsx"


class ImageFileType(enum.Enum):
    """
    Image File Types.

    File types for images for export. These include both raster formats
    (``*.png`` and ``*.tiff``) and vector formats (``*.svg`` and
    ``*.pdf``). The members of this enumerated type have as value the
    string extensions for the respective file types **without** the dot.
    When constructing a filename, the dot must be added.

    Attributes:
        * PNG: Portable Network Graphics (png) image (raster).
        * TIFF: Tagged Image File Format (tiff) image (raster).
        * SVG: Scalable Vector Graphic (svg) image (vector).
        * PDF: Portable Document Format (pdf) file (vector).
    """

    PNG = "png"
    TIFF = "tiff"
    SVG = "svg"
    PDF = "pdf"


def __infer_filetype_from_filename(
    filename: str, file_type_enum: Type[enum.Enum]
) -> Optional[enum.Enum]:
    """
    Infer a file type from a filename.

    This function tries to infer a file type, of the enumerated type
    ``file_type_enum`` from a provided filename by checking the
    extension. If no valid extension is found, ``None`` is returned.
    Otherwise, the determined file type is returned.

    :param filename: string containing the filename.
    :param file_type_enum: enumerated type representing the desired file
        type. This enumerated type should have string values
        representing various file extensions. These values **should
        not** contain a dot.
    :return: ``file_type_enum`` if a valid export filetype is found.
        Otherwise, ``None``.
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
    """
    Infer a ``VectorFileType`` from a filename.

    This function tries to infer a ``VectorFileType`` from a provided
    filename by checking the extension. If no valid extension is found,
    ``None`` is returned. Otherwise, the determined vector type is
    returned.

    :param filename: string containing the filename.
    :return: ``VectorFileType`` if a valid vector filetype is found.
        Otherwise, ``None``.
    """

    vector_file_type = __infer_filetype_from_filename(
        filename=filename, file_type_enum=VectorFileType
    )

    return vector_file_type


def import_vector_field(
    filepath: str,
    default_file_type: VectorFileType = VectorFileType.NPY,
    contains_headers: bool = False,
    sheet_name: Optional[str] = None,
    location_columns: Optional[Sequence[int]] = DEFAULT_LOCATION_COLUMNS,
    component_columns: Sequence[int] = DEFAULT_COMPONENT_COLUMNS,
) -> Optional[np.ndarray]:
    """
    Import a vector field.

    Load a vector field from a file into a NumPy array. For available
    file formats, see ``VectorFileType``. The file type is inferred from
    the filename. If it cannot be inferred, the ``default_file_type`` is
    tried. If the vector field is not valid, then ``None`` is returned.

    :param filepath: File path to the vector field file.
    :param default_file_type: File type to attempt if the type cannot be
        inferred from the filename.
    :param contains_headers: Indicate whether the file contains headers.
        This option is only considered if the vectors are in a CSV or
        Excel file.
    :param sheet_name: Name of the sheet to consider if the vectors are
        in an Excel file.
    :param location_columns: Columns indicating the spatial coordinates
        of the vectors in the order ``(x, y, z)``. If this is set to
        ``None``, the vectors are all assumed to be located at the
        origin. By default, the first three columns are assumed to
        refer to ``(x, y, z)``, respectively.
    :param component_columns: Column indices referring to the vector
        components in the order ``(vx, vy, vz)``. This argument
        **cannot** be ``None``. By default, columns ``(-3, -2, -1)`` are
        assumed to be the ``(vx, vy, vz)``.
    :return: NumPy array containing the vectors. The array has shape
        ``(n, 3)`` or ``(n, 6)``, depending on whether the locations
        are included. The columns correspond to ``(x,y,z)`` coordinates
        of the location (if available), followed by ``(vx, vy, vz)``
        components. If the filetype cannot be properly inferred,
        a value of ``None`` is returned instead.
    """

    # First, infer the file type from the filename
    filetype = __infer_vector_filetype_from_filename(filepath)

    # If inference fails, try the default file type.
    if filetype is None:
        filetype = default_file_type

    if filetype is VectorFileType.NPY:
        try:
            vector_field: np.ndarray = np.load(filepath)
        except (OSError, ValueError):
            # Invalid NumPy array, so return None.
            return None

    # Use Pandas in the other cases
    else:
        header_row: Optional[int] = 0 if contains_headers else None
        try:
            # Reading function depends on whether CSV or Excel
            if filetype is VectorFileType.CSV:
                vector_field_dataframe = pd.read_csv(filepath, header=header_row)
            elif filetype is VectorFileType.EXCEL:
                vector_field_dataframe = pd.read_excel(
                    filepath, sheet_name=sheet_name, header=header_row
                )
            else:
                return None
            vector_field = vector_field_dataframe.to_numpy()
        except (OSError, ValueError):
            return None

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

    return clean_vector_field


def __export_data(
    data: np.ndarray,
    filepath: str | pd.ExcelWriter,
    column_headers: Optional[List] = None,
    indices: Optional[List] = None,
    sheet_name: str = "Sheet1",
    file_type: Optional[VectorFileType] = None,
):
    """
    Export array data to file.

    Export numeric array data to a file with one of the valid file types
    enumerated in ``NumericExportType``. This function will
    automatically infer the filetype if ``None`` is specified for
    ``file_type``. If no valid file type is found, the file type is
    assumed to be CSV.

    :param data: data to write to file. This should be a 2D NumPy array.
    :param filepath: filename where the data should be saved. If
        ``file_type`` is ``None``, this filename is used to infer the
        preferred export type. Alternatively, an ``ExcelWriter`` may be
        passed here if multiple sheets are to be written to a single
        Excel file. If an Excel filename is passed and not an
        ExcelWriter, the contents of the file **will be overwritten**,
        even if there is no sheet with the specified ``sheet_name``.
    :param column_headers: Headers for the column names. If the desired
        file format permits, these will be written in the output file.
    :param indices: indices to use when saving the data. This allows
        naming the rows of the data frame.
    :param sheet_name: If the export file type allows (i.e., the export
        format is ``EXCEL``), this will be used as the name of the
        sheet. This will allow saving multiple pieces of data in the
        same spreadsheet file.
    :param file_type: Instance from ``NumericExportType`` to specify the
        output file type. If ``None`` is specified (or the argument is
        omitted), the file type will be inferred from the filename. If
        no such inference can be performed, the export type will be
        assumed to be CSV.
    :return: ``None``
    """

    if isinstance(filepath, pd.ExcelWriter):
        file_type = VectorFileType.EXCEL

    if file_type is None:
        inferred_filetype = __infer_vector_filetype_from_filename(filename=filepath)

        if inferred_filetype is None:
            inferred_filetype = VectorFileType.CSV

        file_type = inferred_filetype

    file_extension = file_type.value

    if not isinstance(filepath, pd.ExcelWriter) and not filepath.endswith(
        file_extension
    ):
        filepath = f"{filepath}.{file_extension}"

    # Now, we do a different saving procedure depending on the file type
    if file_type is VectorFileType.NPY:
        np.save(filepath, data)
        return

    # For Excel and CSV, we save using Pandas

    # And now, we prepare to save using Pandas
    vector_data_frame = pd.DataFrame(data=data, columns=column_headers, index=indices)

    should_write_indices = indices is not None
    should_write_columns = column_headers is not None

    if file_type is VectorFileType.CSV:
        vector_data_frame.to_csv(
            filepath, sep="\t", index=should_write_indices, header=should_write_columns
        )
        return

    elif file_type is VectorFileType.EXCEL:
        vector_data_frame.to_excel(
            filepath,
            sheet_name=sheet_name,
            index=should_write_indices,
            header=should_write_columns,
        )
        return


def export_vectors_with_orientations(
    vectors: np.ndarray,
    angles: np.ndarray,
    filepath: str | pd.ExcelWriter,
    sheet_name: str = "Sheet1",
    file_type: Optional[VectorFileType] = None,
):
    """
    Export vectors with orientation data.

    Save an array of vectors components, as well as their orientation
    information, to a file. If this file is a CSV or an Excel file, a
    header will be created to name the columns. If the vector array has
    six columns, the first three are assumed to be the locations in
    ``x, y, z`` while the final three are assumed to be the vector
    components in ``x, y, z``. The angles are added on in the order of
    ``phi, theta``.

    :param vectors: NumPy array of shape ``(n, 3)`` or ``(n, 6)``
        containing ``n`` vectors. If there are 6 columns, the first 3
        are assumed to be the location of the vectors in space while
        the last 3 are assumed to be the vector components.
    :param angles: NumPy array of shape ``(n, 2)`` containing the phi
        and theta angles for each vector.
    :param filepath: file path to save the vectors to. If this filename
        has an extension, omit ``file_type`` and this function will
        infer the file type from the extension. Otherwise, the final
        filename will be the value passed here, with the appropriate
        extension appended. This may also be an ``ExcelWriter`` in order
        to support saving multiple sheets to a common Excel file.
    :param sheet_name: name of the sheet if saving to Excel.
    :param file_type: ``NumericExportType`` indicating the desired
        output type. If this in ``None``, the filetype will be inferred
        from the filepath. If no extension is present in either
        argument, the output type will be assumed to be CSV.
    :return: ``None``.
    """

    # Start by concatenating everything
    vectors_with_orientation = np.concatenate([vectors, angles], axis=-1)

    # Generate the column headers
    column_headers = ["Vx", "Vy", "Vz", "PHI", "THETA"]

    has_coordinates = vectors.shape[1] == 6

    if has_coordinates:
        location_headers = ["x", "y", "z"]
        column_headers = location_headers + column_headers

    # sheet_name = "VectorsOrientations"

    __export_data(
        data=vectors_with_orientation,
        filepath=filepath,
        column_headers=column_headers,
        sheet_name=sheet_name,
        file_type=file_type,
    )


def export_one_dimensional_histogram(
    histogram_bins: np.ndarray,
    histogram_values: np.ndarray,
    filepath: str | pd.ExcelWriter,
    bins_header: str = "Bin",
    value_header: str = "Count",
    sheet_name: str = "Sheet1",
    file_type: Optional[VectorFileType] = None,
):
    """
    Export a 1D histogram to a file.

    Save the provided histogram information to a file. The file is
    structured to indicate clearly where each bin starts and ends.
    Therefore, regardless of the file format used, the table will have
    three columns, containing the bin start, bin end and bin value.

    If the ``file_type`` parameter is ``CSV`` or ``EXCEL``, a header row
    will be added. The headers for the first two columns depend on the
    value of ``bins_header``, followed by the respective "_Start" and
    "_End" text. The column containing the bin values is named with
    ``value_header``.

    An ``ExcelWriter`` object can be passed instead of a filename to
    extend an existing Excel sheet that is being written.

    :param histogram_bins: One-dimensional NumPy array of shape
        ``(n + 1, )`` where ``n`` is the number of bins. This array
        contains the boundaries of the histogram.
    :param histogram_values: One-dimensional NumPy array of shape
        ``(n, )`` where ``n`` is the number of bins. This array contains
        the values of each bin of the histogram. These may be raw
        counts, or a more complicated magnitude weighting.
    :param bins_header: text to use for the header of the bin start and
        end columns if saving to CSV or Excel.
    :param value_header: text to use for the header of the histogram
        values column if saving to CSV or Excel.
    :param sheet_name: name of the sheet if saving to Excel.
    :param filepath: string containing path to the output location or
        ``ExcelWriter`` if saving many sheets to the same Excel file.
        If the ``file_type`` is set to ``None``, this filename is used
        to infer the file type. If no file type can be inferred, CSV
        is assumed.
    :param file_type: instance of ``NumericExportType`` indicating the
        desired output file type of the histogram. If this is ``None``,
        the function attempts to infer the file type from the provided
        ``filepath``. If inference fails, the file_type defaults to CSV.
    :return: ``None``.
    """

    histogram_bin_starts = histogram_bins[:-1]
    histogram_bin_ends = histogram_bins[1:]

    complete_histogram_data = np.stack(
        [histogram_bin_starts, histogram_bin_ends, histogram_values], axis=-1
    )

    start_label = f"{bins_header}_Start"
    end_label = f"{bins_header}_End"

    column_headers = [start_label, end_label, value_header]

    __export_data(
        data=complete_histogram_data,
        filepath=filepath,
        column_headers=column_headers,
        sheet_name=sheet_name,
        file_type=file_type,
    )


def export_two_dimensional_histogram(
    histogram_bins: np.ndarray,
    histogram_values: np.ndarray,
    filepath: str | pd.ExcelWriter,
    sheet_name: str = "Sheet1",
    file_type: Optional[VectorFileType] = None,
):
    """
    Export a 2D histogram.

    Save a 2D histogram to file. The histogram is altered to indicate
    clearly where the bins begin and end. The final row and column
    contain zero as there are no values beyond the end of the final bin.
    The histogram must be square. If saving as CSV or EXCEL, the first
    row and first column serve as headers for the table, containing the
    bin start for a given cell. If saving as NPY, the bins are not
    encoded in the file. Instead, a separate file is saved containing
    the histogram bins.

    :param histogram_bins: NumPy array containing the histogram bin
        boundaries. For a histogram of shape ``(n, n)``, this array
        will have shape ``(2, n + 1)``, where ``n`` is the number of
        histogram bins. The zero-indexed bins in axis zero correspond to
        the **rows** of the histogram array, while the one-indexed bins
        correspond to the **columns** in the histogram array.
    :param histogram_values: 2D NumPy array containing the histogram.
        This array has shape ``(n, n)`` where ``n`` is the number of
        histogram bins for each axis.
    :param filepath: string containing path to the output location or
        ``ExcelWriter`` if saving many sheets to the same Excel file.
        If the ``file_type`` is set to ``None``, this filename is used
        to infer the file type. If no file type can be inferred, CSV
        is assumed.
    :param sheet_name: name of the sheet if saving to Excel.
    :param file_type: instance of ``NumericExportType`` indicating the
        desired output file type of the histogram. If this is ``None``,
        the function attempts to infer the file type from the provided
        ``filepath``. If inference fails, the file_type defaults to CSV.
    :return: ``None``.
    """

    # Start by padding the histogram
    padded_histogram = np.pad(histogram_values, ((0, 1), (0, 1)), constant_values=0)

    # Get the bin labels
    row_labels = histogram_bins[0]
    column_labels = histogram_bins[1]

    __export_data(
        data=padded_histogram,
        filepath=filepath,
        column_headers=column_labels,
        indices=row_labels,
        sheet_name=sheet_name,
        file_type=file_type,
    )

    if file_type is VectorFileType.NPY:
        bin_filepath = filepath
        filename_addition = "_histogram_bins.npy"

        if bin_filepath.endswith(".npy"):
            bin_filepath = bin_filepath.replace(".npy", filename_addition)
        else:
            bin_filepath += filename_addition

        np.save(bin_filepath, arr=histogram_bins)
