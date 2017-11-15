import loader
import numpy as np
import metrics
import scaler
from predictions import predict, prediction_error
from matplotlib import pyplot
from sklearn.model_selection import KFold
from hypercubes import compute_distance_rates, compute_error_rates


def main():

    # Hyper cubes
    data, max_dimensions = loader.load_hypercubes()
    compute_error_rates(data, max_dimensions)
    compute_distance_rates(data, max_dimensions)

    # Faces
    data = loader.load_faces()
    x_train, x_test, y_train, y_test = data
    # Computed results
    predicted_data = predict(data)
    print("Error of face prediction: " + str(prediction_error(predicted_data, y_test)))
    alpha_range = range(1, 21)
    alpha_errors = []
    for alpha in alpha_range:
        scaled_data = scaler.scale(data, alpha)
        prediction = predict(scaled_data)
        error = prediction_error(prediction, y_test)
        alpha_errors.append(error)
        print("For scale: " + str(alpha) + " error is: " + str(error))
    pyplot.plot(alpha_range, alpha_errors, 'r-')
    pyplot.plot(alpha_range, alpha_errors, 'bo')
    pyplot.show()
    # Spam
    spam_data = loader.load_spam()
    x_train, x_test, y_train, y_test = spam_data
    # Computed results
    predicted_data = predict(spam_data)
    # Check accuracy of prediction
    print("Error of spam prediction: " + str(prediction_error(predicted_data, y_test)))
    cross_validate(spam_data)




def cross_validate_specific(x, y, folds_number):
    rkf = KFold(n_splits=folds_number, shuffle=True)
    error_rates = []
    for train, test in rkf.split(x, y):
        data = x[train], x[test], y[train], y[test]
        prediction = predict(data)
        error = prediction_error(prediction, y[test])
        error_rates.append(error)
    mean_error = np.array(error_rates).mean()
    return mean_error


def cross_validate(data, folds=5, repeats=10):
    x_train, x_test, y_train, y_test = data
    # Concat test and train data
    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    for i in range(10):
        print("{i}. Cross validation test error {error}".format(i=i + 1, error=cross_validate_specific(x, y, 5)))



if __name__ == '__main__':
    main()
