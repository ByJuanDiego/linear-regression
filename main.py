import csv
import matplotlib.pyplot as plt
from typing import List, Tuple
import json


class LinearModel:

    def __init__(self, data):
        self.data = data
        self.beta = 0
        self.bias = 0

    def eval_point_with_params(self, x):
        return self.beta * x + self.bias

    def predict(self, x):
        return self.eval_point_with_params(x)

    def delta_err_beta(self, loss_funct, data):  # numeric derivative of err with respect to beta
        return loss_funct(self, data)

    def delta_err_bias(self):
        pass

    def calculate_beta(self):
        nummerator = 0
        dennominator = 0

        for point in self.data:
            nummerator += point[0] * point[1]
            dennominator += point[0] ** 2

        self.beta = nummerator / dennominator

    def get_beta(self):
        return self.beta


def mse(model, data):
    accm = 0
    for i in data:  # let dataset be a set of (x,y).
        accm += (model.predict(i[0]) - i[1]) ** 2
    return accm


class DataSet:
    raw_data_file_name = None
    procesed_data_file_name = None

    def __init__(self, csv_filename, json_filename):
        self.raw_data_file_name = csv_filename
        self.procesed_data_file_name = json_filename

    def generate_json_dataset_from_csv(self, independant_variable: str, dependant_variable: str,
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

    def plot(self, y_hat):
        points = self.get_dataset()

        x = [0, 3703895074]
        y = [y_hat(i) for i in x]

        plt.plot(x, y)
        plt.plot(*zip(*points), '.')
        plt.show()


dataset_manager = DataSet("dataset/spotify-2023.csv", "dataset/points.json")
dataset_manager.generate_json_dataset_from_csv("streams", "in_spotify_playlists", 2000, 2023)


model = LinearModel(dataset_manager.get_dataset())
model.calculate_beta()

# dataset_manager.plot(model.eval_point_with_params)

print(model.get_beta())
print(mse(model, dataset_manager.get_dataset()))
