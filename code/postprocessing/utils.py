import os


def read_file(file_path):
    with open(file_path, "r") as f:
        l = f.read().splitlines()
    return l


def load_common_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "helper")
    male_names = read_file(os.path.join(data_path, "male.txt"))
    female_names = read_file(os.path.join(data_path, "female.txt"))
    places = read_file(os.path.join(data_path, "places.txt"))
    return male_names, female_names, places