# Scale input data
def scale(data, scale_factor):
    x_train, x_test, y_train, y_test = data
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[:,-1] *= scale_factor
    x_test[:,-1] *= scale_factor
    return x_train, x_test, y_train, y_test
