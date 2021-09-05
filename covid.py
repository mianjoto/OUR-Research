import numpy as np
import array_driver as ad
from category_handler import us_state_number
from array_driver import colors as c

'''For this one, I will attempt to turn qualitative data into quantitative data using if/else statements'''


'''goals:
    - if you cannot convert the entry into an int, turn the characters of the
    string into the ASCII value representation '''

filename = 'COVID-19_Vaccine_Distribution_Allocations_by_Jurisdiction_-_Pfizer.csv'

'''This will be file-specific file-reading code'''


def create_array():
    array = np.array([], int)
    text_array = []
    with open(filename, 'r') as data:
        index = 0
        _ = data.readline()  # skip the first line, which is variable names
        for line in data:
            split_line = line.split(",")
            for entry in split_line:
                if entry.find('/') == -1:  # if data does not have a date, continue and print
                    if entry in us_state_number:  # if the data is a state
                        entry = int(us_state_number[entry])
                    elif type(entry) is str:
                        text_array.append(entry)
                    array = np.append(array, entry)
                else:  # if the entry is a date, replace without slashes
                    stripped_entry = int(entry.replace("/", ""))
                    # print(f'{entry} turned to {stripped_entry}')
                    array = np.append(array, stripped_entry)
                # if entry.find('\n') == -1:
                #     entry = entry.replace("\n", "")
                print(f'\t\t\t{entry=}, {index=}')  # debugging
                index += 1
        ad.print_array(array)
        width, length = ad.get_width(filename), ad.get_length(filename)
        print(f'{width=}, {length=}')
        print(c.fg.pink, text_array, c.reset)
        array = np.reshape(array, (width, length))
        return array


def alphabet_to_number():
    for i in range(25):
        print("Character for ASCII value", i, "is", chr(i))


alphabet_to_number()
# zero_array = np.empty((ad.get_width(filename), ad.get_length(filename)))
# ad.print_array(zero_array)
filled_array = create_array()
ad.print_array(filled_array)
