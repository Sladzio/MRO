from sklearn.metrics import accuracy_score
import numpy as np
import metrics


def prediction_error(prediction, y_test):
    return 1 - accuracy_score(y_test, prediction)


def predict(data):
    x_train, x_test, y_train, y_test = data
    # Populating array with computed values
    return np.array([predict_specific(x_elem, x_train, y_train, metrics.euclides) for x_elem in x_test])


def predict_specific(x_test_element, x_train, y_train, metric):
    distance = metric(x_train, x_test_element)
    nearest = np.argmin(distance)
    return y_train[nearest]
