import csv
import os.path

from geopy.geocoders import Nominatim
import numpy as np


"""
Getting a bit distracted from the task at hand... now to Scikit-learn
"""


geolocator = Nominatim(user_agent="MIGUEL_OUR_RESEARCH")


def location_to_geocode(location_name: str):
    location = geolocator.geocode(location_name)
    return location.latitude, location.longitude


def load_location_to_geocode_cvs(locations: np.ndarray):
    filename = 'coords.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a+') as file:
        headers = ['location_name', 'coordinates']
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        for location_name in locations:
            # Check if we've already added the city
            for line in file:
                print(f'{line=}')
                if location_name in line:
                    print('found it')
                    continue
            coordinates = location_to_geocode(location_name)
            writer.writerow({'location_name': location_name, 'coordinates': coordinates})

    print('Done!!')


# location_list = np.array(["Charlotte", "Puerto Rico", "Guam"])
print(location_to_geocode("Federal Entities"))
# load_location_to_geocode_cvs(location_list)
