from collections import defaultdict
from scipy.io import loadmat
import numpy as np
import random
from scipy.spatial import distance
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

from functools import wraps, partial
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('{f.__name__} - elapsed time: {time}'.format(f=f, time=end - start))

        return result

    return wrapper


class KMeans:
    def __init__(self, K, X, features_to_image, metrics=distance.euclidean, iterations=1):
        self.K = K
        self.X = X
        self.N = X.shape[0]
        self.clusters = None
        self.centers = []
        self.metrics = metrics
        self.iterations = iterations
        self.prev_centers = []
        self.features_to_image = features_to_image

    def plot_clusters(self):
        directory_name = 'results/k({k})/metric({metric.__name__})'.format(k=self.K, metric=self.metrics)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        for cluster, faces in self.clusters.items():
            figure = plt.figure()
            title = 'Cluster {id}, K - {k}, Metric - {metric.__name__}'.format(id=cluster+1, k=self.K,
                                                                               metric=self.metrics)
            figure.suptitle(title, fontsize=14, fontweight='bold')
            figure.subplots_adjust(top=1)
            image_size = np.ceil(np.sqrt(len(faces))).astype(int)
            for i, vector in enumerate(faces):
                image = self.features_to_image[tuple(vector)]
                subplot = figure.add_subplot(image_size, image_size, i + 1)
                subplot.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
                subplot.axis('off')
            figure.tight_layout()
            figure.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=0.2, hspace=0.2)
            save_path = os.path.join(directory_name, '{}.png'.format(cluster))
            figure.savefig(save_path)
            plt.show()


    def nearest_centroid(self, x_element):
        return np.argmin([self.metrics(center, x_element) for center in self.centers])

    @timing
    def clusterize(self):
        for i in tqdm(range(self.iterations)):
            # if self.check_for_stop():
            #     print("Centers stopped converging at {i}th iteration".format(i=i))
            #     break
            self.clusters = defaultdict(list)
            self.prev_centers = list(self.centers)
            for x_element in self.X:
                self.clusters[self.nearest_centroid(x_element)].append(x_element)
            self.recompute_centers()

        return self.clusters

    def check_for_stop(self):
        return np.array_equiv(np.sort(self.prev_centers, axis=0), np.sort(self.centers, axis=0))

    def recompute_centers(self):
        for cluster, vectors in self.clusters.items():
            self.centers[cluster] = np.mean(vectors, axis=0)

    @timing
    def find_centers(self):
        self.centers = random.choices(self.X, k=self.K)


class KMeans_PlusPlus(KMeans):
    def distances_from_centers(self):
        # Find closest center
        return np.array([min([self.metrics(center, x) for center in self.centers]) ** 2 for x in self.X])

    def find_next_center(self, distances):
        index = np.random.choice(np.arange(len(self.X)), p=distances / distances.sum())
        return self.X[index]

    @timing
    def find_centers(self):
        # Get random element
        self.centers = []
        self.centers.append(random.choice(self.X))
        # Repeat until you find K centers
        while len(self.centers) < self.K:
            # Foreach x data point compute its distance to center
            self.centers.append(self.find_next_center(self.distances_from_centers()))


def load_yale_faces():
    data = loadmat('facesYale.mat')
    images = np.dstack((data['facesTrain'], data['facesTest']))
    images = images.swapaxes(0, 2)
    images = images.swapaxes(1, 2)
    x = np.vstack((data['featuresTrain'], data['featuresTest']))
    y = np.concatenate((data['personTrain'].flatten(), data['personTest'].flatten()))
    return x, y, images


def run_kmeans():
    print("K-means")
    x_data, y_data, images = load_yale_faces()
    features_to_img = {tuple(features): image for features, image in zip(x_data, images)}
    k_means = KMeans(2, x_data, features_to_image=features_to_img, iterations=100)
    k_means.find_centers()
    k_means.clusterize()
    k_means.plot_clusters()


def run_kmeans_plus_plus():
    print("K-means Plus Plus")
    x_data, y_data, images = load_yale_faces()
    features_to_img = {tuple(features): image for features, image in zip(x_data, images)}
    k_means = KMeans(2, x_data, features_to_image=features_to_img, iterations=100)
    k_means.find_centers()
    k_means.clusterize()
    k_means.plot_clusters()


def mahalanobis_experiment():
    x_data, y_data, images = load_yale_faces()
    features_to_img = {tuple(features): image for features, image in zip(x_data, images)}
    vi = np.linalg.inv(np.cov(x_data.T))

    mahalanobis_distance = partial(distance.mahalanobis, VI=vi)
    mahalanobis_distance.__name__ = 'mahalanobis'
    for k in [2, 5, 8, 10]:
        for metric in [distance.euclidean, mahalanobis_distance]:
            k_means = KMeans_PlusPlus(k, x_data, features_to_image=features_to_img, iterations=100, metrics=metric)
            k_means.find_centers()
            k_means.clusterize()
            k_means.plot_clusters()
