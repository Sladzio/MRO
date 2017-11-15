from predictions import prediction_error, predict
import metrics
from matplotlib import pyplot
import numpy as np


def compute_error_rates(data, max_dimensions):
    error_rates = []
    dimensions = range(1, max_dimensions + 1)
    for n in dimensions:
        nth_data = get_nth_dimension(data, n)
        x_train, x_test, y_train, y_test = nth_data
        prediction = predict(nth_data)
        error = prediction_error(prediction, y_test)
        error_rates.append(error)
    pyplot.plot(dimensions, error_rates, 'b-')
    pyplot.plot(dimensions, error_rates, 'g^')
    pyplot.show()


def compute_class_distances(x_test_element, y_test_element, x_train, y_train, ):
    # Get train class
    x_train_class = x_train[y_train == y_test_element]
    # Get train opposite class
    x_train_opposite_class = x_train[y_train != y_test_element]
    distance = metrics.euclides(x_test_element, x_train_class).min()
    opposite_distance = metrics.euclides(x_test_element, x_train_opposite_class).min()
    return distance, opposite_distance


def compute_distance_rates(data, max_dimensions):
    class_distances = []
    opposite_class_distances = []
    dimensions = range(1, max_dimensions + 1)
    for n in dimensions:
        nth_data = get_nth_dimension(data, n)
        x_train, x_test, y_train, y_test = nth_data
        distances = np.array([compute_class_distances(x_test_element, y_test_element, x_train, y_train)
                              for x_test_element, y_test_element in zip(x_test, y_test)])

        # Compute mean
        mean_class_distance, mean_opposite_class_distance = distances.mean(axis=0)
        class_distances.append(mean_class_distance)
        opposite_class_distances.append(mean_opposite_class_distance)

    class_distance_ratios = np.array(class_distances) / np.array(opposite_class_distances)

    pyplot.plot(dimensions, class_distances, 'r-', label='Class distances')
    pyplot.plot(dimensions, opposite_class_distances, 'g-', label='Opposite class distances')
    pyplot.plot(dimensions, class_distance_ratios, 'b-', label='Class distance ratios')
    pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=2, mode="expand", borderaxespad=0.)
    pyplot.show()


# Get data in nth dimension
def get_nth_dimension(data, dimensions_number):
    x_train, x_test, y_train, y_test = data
    dimension_index = dimensions_number - 1
    x_nth_train = x_train[dimension_index]
    x_nth_test = x_test[dimension_index]
    y_nth_train = y_train[dimension_index].flatten()
    y_nth_test = y_test[dimension_index].flatten()
    return x_nth_train, x_nth_test, y_nth_train, y_nth_test
