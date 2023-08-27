import csv
import matplotlib.pyplot as plt
from typing import List, Tuple
import json


class LinearModel:
    def __init__(self, beta=0, bias=0):
        self.beta = 0
        self.bias = 0

    @staticmethod
    def eval_point_with_params(self, x, beta, bias):
        return beta * x + bias

    def predict(self, x):
        return self.eval_point_with_params(x, self.beta, self.bias)

    def delta_err_beta(self, loss_funct, dataset):  # numeric derivative of err with respect to beta
        LinearModel()
        return loss_funct(self, dataset)

    def delta_err_bias(self):
        pass


def mse(model, dataset):
    accm = 0
    for i in dataset:  # let dataset be a set of (x,y).
        accm += (model.predict(i[0]) - i[1]) ** 2
    return accm


class DataSet:
    raw_data_file_name = None
    procesed_data_file_name = None

    def __init__(self, csv_filename, json_filename):
        self.raw_data_file_name = csv_filename
        self.procesed_data_file_name = json_filename

    def generate_dataset(self, independant_variable: str, dependant_variable: str,
                         min_year: int = 2000, max_year: int = 2023):

        with open(self.raw_data_file_name, 'r') as csv_file, open(self.procesed_data_file_name, 'w') as json_file:
            spamreader = csv.DictReader(csv_file, delimiter=',')
            points: List[Tuple[int, int]] = []

            for row in spamreader:
                if min_year <= int(row['released_year']) <= max_year:
                    points.append((int(row[independant_variable]), int(row[dependant_variable])))

            points.sort(key=lambda x: x[0])
            json.dump(points, json_file, indent=4)

    def get_dataset(self) -> List[Tuple[int, int]]:
        with open(self.procesed_data_file_name, 'r') as json_file:
            return json.load(json_file)

    def plot(self):
        points = self.get_dataset()
        plt.plot(*zip(*points), '.')
        plt.show()


dataset = DataSet("dataset/spotify-2023.csv", "dataset/points.json")
dataset.generate_dataset("streams", "in_spotify_playlists", 2000, 2023)
dataset.plot()
