import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

def bike():
    data = pd.read_csv("day.csv")
    non_holiday = data[data["holiday"] == 0][["casual", "registered"]]
    holiday = data[data["holiday"] == 1][["casual", "registered"]]
    print("Rentals on a holiday:")
    print(holiday.mean())
    print()

    print("Rentals on a non-holiday:")
    print(non_holiday.mean())
    print()

    temp = data["temp"] * (39 - (-8)) + (-8)
    print("Maximum temperature: ", temp.max())
    print("Minimum temperature: ", temp.min())
    print()

    print("Dates on which more casual than registered useres were present")
    print(data[data["registered"] < data["casual"]][["dteday"]])
    print()

    plt.figure()
    plt.scatter(temp, data["registered"], color='b')
    plt.scatter(temp, data["casual"], color='r')
    return

def titanic():
    data = pd.read_csv("titanic.csv")

    print("Survival rate: ", len(data[data["Survived"] == 1]) / len(data))
    print("Survival rate of male passengers: ",
    len(data[(data["Survived"] == 1) & (data["Sex"] == "male")])
    / len(data[data["Sex"] == "male"]))

    print("Survival rate of female passengers: ",
    len(data[(data["Survived"] == 1) & (data["Sex"] == "female")])
    / len(data[data["Sex"] == "female"]))
    print("Average fare paid by survivors: ", data[data["Survived"] == 1]
    ["Fare"].mean())
    print("Average fare paid by casulties: ", data[data["Survived"] == 0]
    ["Fare"].mean())
    print()

    return


def generate_data():
    no_of_clusters = 3
    cluster_mean = np.random.rand(no_of_clusters, 2)
    data = np.array([[]])
    target = np.array([[]], dtype='int')
    points_per_cluster = 100
    sigma = 0.1

    for i in range(no_of_clusters):
        noise = sigma * np.random.randn(points_per_cluster, 2)
        cluster = cluster_mean[i, :] + noise
        data = np.append(data, cluster).reshape((i + 1) * points_per_cluster, 2)
        target = np.append(target, [i] * points_per_cluster)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=target)

    return data, target


def decision_boundaries(data, target):
    clf = svm.SVC()
    clf.fit(data, target)
    x_min = min(data[:, 0])
    x_max = max(data[:, 0])
    y_min = min(data[:, 1])
    y_max = max(data[:, 1])
    granularity = 0.01

    x, y = np.meshgrid(np.arange(x_min, x_max, granularity), np.arange(y_min,
    y_max, granularity))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    prediction = clf.predict(xy)
    prediction = prediction.reshape(x.shape)
    plt.figure()

    plt.imshow(prediction, extent=(x_min, x_max, y_min, y_max), alpha=0.4,
    origin="lower")

    plt.scatter(data[:, 0], data[:, 1], c=target)
    return

def main():
    plt.close("all")
    bike()
    titanic()
    data, target = generate_data()
    decision_boundaries(data, target)
    plt.show()

main()