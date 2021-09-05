import numpy as np
from color_print import ColorPrint as cp


def get_length(filepath):
    length = -1  # Do not include the first variable line
    with open(filepath, 'r') as data:
        length = sum(1 for line in open(filepath))
    return length


def get_width(filepath):
    with open(filepath, 'r') as data:
        width = len(
            data.readline().split(","))  # get the width of the first line
    return width


# Returns a tuple of rows, columns
def get_shape(filepath):
    return get_length(filepath), get_width(filepath)


def remove_slash(s: str):
    return s.replace('/', '')


def remove_newline(s: str):
    return s.replace('\n', '')


def str_to_ascii(s: str):
    ascii_str = ""
    for letter in s:
        ascii_str += str(ord(letter))
    return float(ascii_str)


def ascii_to_str(i: float):
    num_of_chars = int(len(str(i)))
    i_string = str(i)
    ascii_str = ""
    for index in range(3, num_of_chars + 1, 3):
        ascii_str += chr(int(i_string[
                             index - 3: index]))  # Convert every 3 numbers
        # into the ASCII letter/symbol
    return ascii_str


# Cleans and converts a data entry into a float or ASCII float representing
# the value
def type_handler(entry: str):
    slash = "/"
    newline = "\n"

    # If the entry has a slash/newline, remove the slash/newline
    entry = remove_slash(entry) if slash in entry else entry
    entry = remove_newline(entry) if newline in entry else entry

    # Convert entry to int; if entry is a string, then convert to ASCII
    try:
        entry = float(entry)
    except ValueError:
        entry = str_to_ascii(entry)

    return entry


def print_np_array(np_array: np.ndarray):
    length, width = np_array.shape[0] - 1, np_array.shape[1]
    for i in np.arange(length):
        line = "["
        for j in np.arange(width):
            line += str(np_array[i, j])
            if j != width - 1:
                line += ', '
        line += "]"
        print(line)
    cp.print_green('successfully printed the array!!')


def create_array_from_csv(filepath: str):
    dimension = get_shape(filepath)
    np_array = np.zeros(dimension)
    cp.print_green(f'Created a NumPy array of shape {np_array.shape}')
    with open(filepath, 'r') as data:
        _ = data.readline()  # Skip variable line
        column_index, row_index = 0, 0  # Since 1st line is skipped,
        # index is incremented by 1
        for line in data:
            split_entries = line.split(",")
            column_index = 0
            for entry in split_entries:  # Iterate through each value
                entry = type_handler(entry)
                np_array[row_index, column_index] = entry
                column_index += 1
            row_index += 1
    return np_array


def create_and_print_array_from_csv(filepath: str):
    array = create_array_from_csv(filepath)
    print_np_array(array)
    return array
