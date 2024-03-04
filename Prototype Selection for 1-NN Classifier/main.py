# Step 1: Importing Necessary Libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import mnist
from scipy.spatial import distance
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt
import concurrent.futures
import random
import json
import time
import warnings
warnings.filterwarnings("ignore")
# Define Parameters
M = [20000, 10000, 5000, 1000, 500, 100]  # Number of samples to select
num_seeds = 10  # Number of different seeds for sampling
# M = [200]  # Number of samples to select
# num_seeds = 2  # Number of different seeds for sampling
K = 1

# Load MNIST Dataset using Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0


class BasePrototype:
    def __init__(self, num_samples, seed):
        self.num_samples = num_samples
        self.seed = seed

    def get_samples(self):
        pass

    def get_result(self):
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(self.X_train_sampled, self.y_train_sampled)
        self.y_pred = knn.predict(X_test)
        return self.y_pred

    def analysis(self, full=False):
        accuracy = accuracy_score(y_test, self.y_pred)
        if full:
            print("Accuracy:", accuracy)
            print("Classification Report:\n", classification_report(y_test, self.y_pred))
        return accuracy

    def all(self, full=False):
        self.get_samples()
        self.get_result()
        return self.analysis(full)


class NoPrototype(BasePrototype):
    def get_samples(self):
        self.X_train_sampled = X_train
        self.y_train_sampled = y_train


class RandomSelectionPrototype(BasePrototype):
    def get_samples(self):
        np.random.seed(self.seed)
        indices = np.random.choice(X_train.shape[0], self.num_samples, replace=False)
        self.X_train_sampled = X_train[indices]
        self.y_train_sampled = y_train[indices]


def process_class(class_label, num_samples, seed):
    # print("K-Means of class {} started".format(class_label))
    # Isolate data for the current class
    class_data = X_train[y_train == class_label]

    # Apply k-means
    kmeans = KMeans(n_clusters=num_samples, random_state=seed)
    kmeans.fit(class_data)
    centroids = kmeans.cluster_centers_

    # Find nearest samples to centroids
    nearest_samples = []
    for centroid in centroids:
        distances = distance.cdist([centroid], class_data, 'euclidean')
        nearest_index = np.argmin(distances)
        nearest_samples.append((class_data[nearest_index], class_label))

    # print("K-Means of class {} finished".format(class_label))
    return nearest_samples


class KMeansPrototype(BasePrototype):
    # def assign_num_prototypes(self):
    #     pass

    def get_samples(self):
        sampled_training_data = []

        # print("Starting Parallel")
        # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #     futures = [executor.submit(process_class, class_label, self.num_samples_dict[class_label], self.seed)
        #                for class_label in np.unique(y_train)]
        #     for future in concurrent.futures.as_completed(futures):
        #         sampled_training_data.extend(future.result())

        for class_label in np.unique(y_train):
            sampled_training_data.extend(process_class(class_label, self.num_samples_dict[class_label], self.seed))

        X_train_sampled, y_train_sampled = zip(*sampled_training_data)
        self.X_train_sampled = np.array(X_train_sampled)
        self.y_train_sampled = np.array(y_train_sampled)
        # self.X_train_sampled = []
        # self.y_train_sampled = []
        #
        # for class_label in np.unique(y_train):
        #     # Isolate data for the current class
        #     class_data = X_train[y_train == class_label]
        #
        #     # Apply k-means
        #     kmeans = KMeans(n_clusters=self.num_samples_dict[class_label], random_state=self.seed)
        #     kmeans.fit(class_data)
        #     print("K-Means of class {}".format(class_label))
        #     centroids = kmeans.cluster_centers_
        #
        #     # Find nearest samples to centroids
        #     for centroid in centroids:
        #         distances = distance.cdist([centroid], class_data, 'euclidean')
        #         nearest_index = np.argmin(distances)
        #         self.X_train_sampled.append(class_data[nearest_index])
        #         self.y_train_sampled.append(class_label)
        #
        # self.X_train_sampled = np.array(self.X_train_sampled)
        # self.y_train_sampled = np.array(self.y_train_sampled)

    def all(self, full=False):
        self.assign_num_prototypes()
        self.get_samples()
        self.get_result()
        return self.analysis(full)


class RandomKMeansPrototype(KMeansPrototype):
    def get_samples(self):
        self.X_train_sampled = []
        self.y_train_sampled = []

        kmeans = KMeans(n_clusters=self.num_samples, random_state=self.seed)
        kmeans.fit(X_train)
        # print("K-Means of random")
        centroids = kmeans.cluster_centers_

        # Find nearest samples to centroids
        for centroid in centroids:
            distances = distance.cdist([centroid], X_train, 'euclidean')
            nearest_index = np.argmin(distances)
            self.X_train_sampled.append(X_train[nearest_index])
            self.y_train_sampled.append(y_train[nearest_index])

        self.X_train_sampled = np.array(self.X_train_sampled)
        self.y_train_sampled = np.array(self.y_train_sampled)

    def all(self, full=False):
        self.get_samples()
        self.get_result()
        return self.analysis(full)


class BalancedSampler:
    def assign_num_prototypes(self):
        num_classes = len(np.unique(y_train))
        quotient = self.num_samples // num_classes
        remainder = self.num_samples - num_classes * quotient
        num_samples_per_class = [quotient + (_ < remainder) for _ in range(num_classes)]
        self.num_samples_dict = dict()
        for i, class_label in enumerate(np.unique(y_train)):
            self.num_samples_dict[class_label] = num_samples_per_class[i]


class WeightedSampler:
    def assign_num_prototypes(self):
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_ratios = np.floor(class_counts / y_train.shape[0] * self.num_samples).astype(int)
        remainder = self.num_samples - class_ratios.sum()
        self.num_samples_dict = dict()
        for i, (class_label, class_ratio) in enumerate(zip(unique_classes, class_ratios)):
            self.num_samples_dict[class_label] = class_ratio + (i < remainder)


class BalancedKMeansPrototype(KMeansPrototype, BalancedSampler):
    pass


class WeightedKMeansPrototype(KMeansPrototype, WeightedSampler):
    pass


class NearMissPrototype(BasePrototype):
    def __init__(self, *args, **kwargs):
        self.nearmiss_type = kwargs['nearmiss_type']
        kwargs.pop('nearmiss_type')
        super().__init__(*args, **kwargs)

    # def assign_num_prototypes(self):
    #     pass

    def get_samples(self):
        nearmiss = NearMiss(sampling_strategy=self.num_samples_dict, version=self.nearmiss_type, n_neighbors=3,
                            n_neighbors_ver3=3)
        self.X_train_sampled, self.y_train_sampled = nearmiss.fit_resample(X_train, y_train)

    def all(self, full=False):
        self.assign_num_prototypes()
        self.get_samples()
        self.get_result()
        return self.analysis(full)


class BalancedNearMissPrototype(NearMissPrototype, BalancedSampler):
    pass


class WeightedNearMissPrototype(NearMissPrototype, WeightedSampler):
    pass


class WeightedMixedPrototype(BasePrototype):
    def __init__(self, *args, **kwargs):
        self.kmeans_ratio = kwargs['kmeans_ratio']
        kwargs.pop('kmeans_ratio')
        self.nearmiss_type = kwargs['nearmiss_type']
        kwargs.pop('nearmiss_type')
        super().__init__(*args, **kwargs)
        self.num_samples_kmeans = int(self.num_samples * self.kmeans_ratio)
        self.num_samples_nearmiss = self.num_samples - self.num_samples_kmeans
        self.nearmiss_prototype = WeightedNearMissPrototype(self.num_samples_nearmiss, self.seed,
                                                            nearmiss_type=self.nearmiss_type)
        self.kmeans_prototype = WeightedKMeansPrototype(self.num_samples_kmeans, self.seed)

    def assign_num_prototypes(self):
        self.nearmiss_prototype.assign_num_prototypes()
        self.kmeans_prototype.assign_num_prototypes()

    def get_samples(self):
        self.nearmiss_prototype.get_samples()
        self.kmeans_prototype.get_samples()
        self.X_train_sampled = np.concatenate(
            (self.nearmiss_prototype.X_train_sampled, self.kmeans_prototype.X_train_sampled))
        self.y_train_sampled = np.concatenate(
            (self.nearmiss_prototype.y_train_sampled, self.kmeans_prototype.y_train_sampled))

    def all(self, full=False):
        self.assign_num_prototypes()
        self.get_samples()
        # self.kmeans_prototype.get_result()
        # self.kmeans_prototype.analysis(full)
        # print('\nabove kmeans\n--------------------------\nbelow nearmiss\n')
        self.get_result()
        return self.analysis(full)


works = {
    'RandomSelectionPrototype': lambda num_samples, seed:
    RandomSelectionPrototype(num_samples, seed).all(),
    # 'RandomKMeansPrototype': lambda num_samples, seed:
    # RandomKMeansPrototype(num_samples, seed).all(),
    'BalancedKMeansPrototype': lambda num_samples, seed:
    BalancedKMeansPrototype(num_samples, seed).all(),
    'WeightedKMeansPrototype': lambda num_samples, seed:
    WeightedKMeansPrototype(num_samples, seed).all(),
    'BalancedNearMissPrototype': lambda num_samples, seed:
    BalancedNearMissPrototype(num_samples, seed, nearmiss_type=1).all(),
    'WeightedNearMissPrototype': lambda num_samples, seed:
    WeightedNearMissPrototype(num_samples, seed, nearmiss_type=1).all(),
    'WeightedMixedPrototype': lambda num_samples, seed:
    WeightedMixedPrototype(num_samples, seed, nearmiss_type=1, kmeans_ratio=0.85).all()
}


def test():
    print(works['WeightedMixedPrototype'](1000, 100))
    # WeightedMixedPrototype(10000, 100, nearmiss_type=1, kmeans_ratio=0.85).all(full=True)
    # WeightedNearMissPrototype(10000, 100, nearmiss_type=1).all(full=True)
    # RandomSelectionPrototype(10000, 100).all(full=True)
    # WeightedKMeansPrototype(10000, 100).all(full=True)
    # NoPrototype(1000, 100).all(full=True)


def plot(accuracies, std_deviation):
    # Optional: Plotting the Accuracies with Error Bars
    plt.errorbar(range(num_seeds), accuracies, yerr=std_deviation, fmt='o')
    plt.title("Accuracy with Error Bars for Different Training Seeds")
    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.show()


def main():
    result_dict = dict()
    print(f"Using num_seeds={num_seeds}")

    start_time = time.time()
    baseline = NoPrototype(0, 0).all()
    end_time = time.time()
    avg_time = end_time - start_time
    print(f"Baseline, Mean Accuracy: {baseline}, Average Time: {avg_time}")
    result_dict['Baseline'] = (baseline, -1, avg_time)

    for work in works:
        result_dict[work] = dict()
        for num_samples in M:
            random_seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
            accuracies = []
            start_time = time.time()

            # Step 2: Random Sample Selection and Training
            for seed in random_seeds:
                accuracies.append(works[work](num_samples, seed))

            # Step 4: Calculate Accuracy and Error Bars
            mean_accuracy = np.mean(accuracies)
            std_deviation = np.std(accuracies)
            end_time = time.time()
            avg_time = (end_time - start_time)/len(random_seeds)

            print(f"{work}, M: {num_samples}, Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_deviation}, "
                  f"Average Time: {avg_time}")
            result_dict[work][num_samples] = (mean_accuracy, std_deviation, avg_time)

        print('----------------------------------------')
        print(json.dumps(result_dict))
        print('----------------------------------------')


if __name__ == '__main__':
    main()
