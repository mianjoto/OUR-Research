import np_array_driver as ad
from geopy.geocoders import Nominatim

import numpy as np

'''KNOWN ISSUES and GOALS: 
    - if the an entry's first letter's ASCII number 
    representation begins with 0, the 0 will be omitted, which will mess up the 
    syntax of the translation from ASCII back to English (see test() function in 
    this file, should print out Wisconsin but does not since it misses the first 
    0 
    
    - must fully delete covid.py and singapore.py 
    
    ### FIXED ### - the last row is not being filled... gotta fix that 
    
    '''


covid_data = 'COVID-19_Vaccine_Distribution_Allocations_by_Jurisdiction_' \
             '-_Pfizer.csv '
singapore_data = 'singapore-residents-by-ethnic-group-and-sex-end-june-annual' \
                 '.csv'


def main():
    pass
    # data = ad.create_array_from_csv(covid_data)
    # ad.print_np_array(data)
    # print(f'{ad.get_length(covid_data)=}')


def test():
    print()


if __name__ == "__main__":
    test()
    main()
