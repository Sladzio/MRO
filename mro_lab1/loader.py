from scipy import io as io


def load_faces():
    mat = io.loadmat('facesYale.mat')
    x_train = mat['featuresTrain']
    x_test = mat['featuresTest']
    y_train = mat['personTrain'].flatten()
    y_test = mat['personTest'].flatten()
    return x_train, x_test, y_train, y_test


def load_spam():
    mat = io.loadmat('spambase.mat')
    x_train = mat['featuresTrain']
    x_test = mat['featuresTest']
    y_train = mat['classesTrain'].flatten()
    y_test = mat['classesTest'].flatten()
    return x_train, x_test, y_train, y_test


def load_hypercubes():
    mat = io.loadmat('multiDimHypercubes.mat')
    x_train = mat['featuresTrain'][0]
    x_test = mat['featuresTest'][0]
    y_train = mat['classesTrain'][0]
    y_test = mat['classesTest'][0]

    max_dimensions = mat['maxDim'][0][0]
    return (x_train, x_test, y_train, y_test), max_dimensions
