import numpy as np


class colors:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'


def get_length(filepath):
    with open(filepath, 'r') as data:
        length = 0
        for _ in data:
            length += 1
    print(f'{length=}')
    return length


def get_width(filepath):
    with open(filepath, 'r') as data:
        width = len(data.readline().split(","))  # get the width of the first line
        print(f'{width=}')
        return width


def print_array(np_array):
    length = len(np_array)
    for i in np.arange(0, length):
        print(np_array[i])


def create_array(filepath):
    array = np.empty((get_width(filepath), get_length(filepath)))
    with open(filepath, 'r') as data:
        index = 0
        for line in data:
            split_line = line.split(",")
            for entry in split_line:
                # print(f'\t{entry=}')  # debugging
                np.append(index, entry)
                index += 1
    return array


def create_and_print_array(filepath):
    array = create_array(filepath)
    print_array(array)
    return array
