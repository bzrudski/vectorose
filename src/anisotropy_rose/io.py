"""
Anisotropy Rose - I/O Operations

Joseph Deering, Benjamin Rudski
2023

This module provides the ability to load vector fields from file and
save vector fields and anisotropy histograms.

"""

import enum
import os
from typing import Optional, Type, List

import numpy as np
import pandas as pd


class NumericExportType(enum.Enum):
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


def __infer_filetype_from_filename(filename: str, file_type_enum: Type[enum.Enum]) -> Optional[enum.Enum]:
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
    cleaned_extension = extension.lstrip('.')

    try:
        # Try to get the file type based on the extension.
        file_type = file_type_enum(cleaned_extension)
    except ValueError:
        # Otherwise, no filetype found.
        file_type = None

    return file_type


def __infer_numeric_filetype_from_filename(filename: str) -> Optional[NumericExportType]:
    """
    Infer a ``NumericExportType`` from a filename.

    This function tries to infer a ``NumericExportType`` from a provided
    filename by checking the extension. If no valid extension is found,
    ``None`` is returned. Otherwise, the determined export type is
    returned.

    :param filename: string containing the filename.
    :return: ``NumericExportType`` if a valid export filetype is found.
        Otherwise, ``None``.
    """

    export_file_type = __infer_filetype_from_filename(filename=filename, file_type_enum=NumericExportType)

    return export_file_type


def __export_data(data: np.ndarray, filepath: str | pd.ExcelWriter,
                  column_headers: Optional[List] = None,
                  indices: Optional[List] = None,
                  sheet_name: str = "Sheet1",
                  file_type: Optional[NumericExportType] = None):
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
        file_type = NumericExportType.EXCEL

    if file_type is None:
        inferred_filetype = __infer_numeric_filetype_from_filename(filename=filepath)

        if inferred_filetype is None:
            inferred_filetype = NumericExportType.CSV

        file_type = inferred_filetype

    file_extension = file_type.value

    if not isinstance(filepath, pd.ExcelWriter) and not filepath.endswith(file_extension):
        filepath = f"{filepath}.{file_extension}"

    # Now, we do a different saving procedure depending on the file type
    if file_type is NumericExportType.NPY:
        np.save(filepath, data)
        return

    # For Excel and CSV, we save using Pandas

    # And now, we prepare to save using Pandas
    vector_data_frame = pd.DataFrame(data=data, columns=column_headers, index=indices)

    should_write_indices = indices is not None
    should_write_columns = column_headers is not None

    if file_type is NumericExportType.CSV:
        vector_data_frame.to_csv(filepath, sep="\t", index=should_write_indices, header=should_write_columns)
        return

    elif file_type is NumericExportType.EXCEL:
        vector_data_frame.to_excel(filepath, sheet_name=sheet_name, index=should_write_indices,
                                   header=should_write_columns)
        return


def export_vectors_with_orientations(vectors: np.ndarray, angles: np.ndarray, filepath: str | pd.ExcelWriter,
                                     sheet_name: str = "Sheet1",
                                     file_type: Optional[NumericExportType] = None):
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

    __export_data(data=vectors_with_orientation,
                  filepath=filepath,
                  column_headers=column_headers,
                  sheet_name=sheet_name,
                  file_type=file_type)


def export_one_dimensional_histogram(histogram_bins: np.ndarray, histogram_values: np.ndarray,
                                     filepath: str | pd.ExcelWriter, bins_header: str = "Bin",
                                     value_header: str = "Count", sheet_name: str = "Sheet1",
                                     file_type: Optional[NumericExportType] = None):
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

    complete_histogram_data = np.stack(histogram_bin_starts, histogram_bin_ends, histogram_values, axis=-1)

    start_label = f"{bins_header}_Start"
    end_label = f"{bins_header}_End"

    column_headers = [start_label, end_label, value_header]

    __export_data(data=complete_histogram_data,
                  filepath=filepath,
                  column_headers=column_headers,
                  sheet_name=sheet_name,
                  file_type=file_type)


def export_two_dimensional_histogram(histogram_bins: np.ndarray, histogram_values: np.ndarray,
                                     filepath: str | pd.ExcelWriter, sheet_name: str = "Sheet1",
                                     file_type: Optional[NumericExportType] = None):
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

    __export_data(data=padded_histogram, filepath=filepath, column_headers=column_labels, indices=row_labels,
                  sheet_name=sheet_name, file_type=file_type)

    if file_type is NumericExportType.NPY:
        bin_filepath = filepath
        filename_addition = "_histogram_bins.npy"

        if bin_filepath.endswith(".npy"):
            bin_filepath = bin_filepath.replace(".npy", filename_addition)
        else:
            bin_filepath += filename_addition

        np.save(bin_filepath, arr=histogram_bins)
