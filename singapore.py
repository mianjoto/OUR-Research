import np_array_driver

'''DEPRECIATED: The following note is old and the Python file will be deleted 
soon:

The goal here was to load some demographic information from a Singapore 
census and be able to access each entry. I now know that you cannot mix 
datatypes in NumPy arrays. If I were to reshape the data to only include 
numbers, I would be able to do this, however, since I cannot, I will not be 
able to use this dataset to test and practice NumPy arrays with. '''

filepath = 'singapore-residents-by-ethnic-group-and-sex-end-june-annual.csv'

# the code below will run on import
np_array_driver.create_and_print_array(filepath)
