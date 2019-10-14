from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

## A helper function
# Converts the given string list to float list.
def convert_to_float(l):
    return [float(i) for i in l]

## Parse data
# Create two numpy arrays for data and the labels
def parse_data(data):
    features = []
    labels = []
    # Remove the new line character at the end.
    lines = [l.replace('\n', '') for l in data]
    # For each sample ...
    for line in lines:
        # Fetch the label
        label = int(line.split(',')[-1])
        # Fetch the features
        feature = convert_to_float(line.split(',')[1:-1])
        # Append the sample feature to the list
        features.append(feature)
        # Append the sample label to the list 
        labels.append(label)
    
    # Convert python lists to numpy arrays.
    features = np.array(features)
    labels = np.array(labels)
    # Return features and the corresponding labels.
    return features, labels

## Load data
def load_data(filename="../Data/glass.data"):
    f = open(filename)
    return parse_data(f.readlines())

# Performs a 5-fold cross validation.
# X : features
# Y : labels
# kernel : kernel type
# shape : one vs. one or one vs. all approach
# weights : default is 'None' which corresponds that all weights are 1.
# .........            'balanced' corresponds to the weights which is inversely proportional weights.
# c : penalty parameter of the error term.
# degree : a hyperparameter for polynomial kernel, default is 3.
# gamma : kernel coefficient for rbf, polynomial, and sigmoid. Default is 'scale'
# ........ which uses 1/(number_of_features * X.var()) as the value of gamma.
def _svm_(X, Y, kernel, shape, weights=None, c=1.0, degree=3, gamma='scale'):
    # SVC is a class defined in sklearn.SVM module.
    # More information can be found in below link.
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    clf = SVC(C=c, kernel=kernel, degree=degree, decision_function_shape=shape, gamma=gamma, class_weight=weights)
    
    # Perform 5-fold cross validation.
    
    # cross_val_score is a function defined in sklearn.model_selection module. 
    # It evaluates a score by cross-validation. 
    # More information can be found in below link.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html 
    # Folds are created by StratifiedKFold.
    # For more information about StrafiedKFold, please see below documentation.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    scores = cross_val_score(clf, X, Y, cv=5)
    return scores

# Performs the svm evaluations for different kernels and weigths.
def default_kernel_evaluations(X, Y, shape, weights=None, plot=True, title=None):
    # Linear kernel
    linear_ovo = _svm_(X, Y, 'linear', shape=shape, weights=weights)

    # RBF kernel
    rbf_ovo = _svm_(X, Y, 'rbf', shape=shape, weights=weights)
    # print (rbf_ovo)

    # Polynomial kernel
    poly_ovo = _svm_(X, Y, 'poly', shape=shape, weights=weights)

    # Sigmoid kernel
    sigmoid_ovo = _svm_(X, Y, 'sigmoid', shape=shape, weights=weights)

    if plot:
        x = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']
        y = [linear_ovo.mean(), rbf_ovo.mean(), poly_ovo.mean(), sigmoid_ovo.mean()]
        pos = [1,2,3,4]
        plt.bar(pos, y, align='center', alpha=0.5)
        plt.xticks(pos, x)
        plt.ylabel('Mean Accuracy')
        plt.xlabel('Kernel Methods')
        plt.title(title)
        plt.show()

def alter_penalty_parameter(X, Y, kernel='linear', shape='ovo', print_report=False):
    scores = []
    if print_report:
        print('#######################################################################')
        print ('Altering penalty parameter of error term.')
        print ("Kernel type : {0}".format(kernel))
        print ("Decision method : {0}".format(shape))
    for c in range(1,5):
        score = _svm_(X, Y, kernel=kernel, shape=shape, c=c)
        scores.append((score.mean(), score.var()))
        if print_report:
            print("C = {0} --> Accuracy: {1} +- {2}".format(float(c), score.mean(), score.var()))
    return scores

def alter_gamma(X, Y, kernel='linear', shape='ovo', print_report=False):
    scores = []
    if print_report:
        print('#######################################################################')
        print ('Altering gamma.')
        print ("Kernel type : {0}".format(kernel))
        print ("Decision method : {0}".format(shape))
    for g in range(1,5):
        score = _svm_(X, Y, kernel=kernel, shape=shape, gamma=g)
        scores.append((score.mean(), score.var()))
        if print_report:
            print("Gamma = {0} --> Accuracy: {1} +- {2}".format(float(g), score.mean(), score.var()))
    return scores


def alter_poly_degree(X, Y, print_report=False):
    scores = []
    if print_report:
        print('#######################################################################')
        print ('Altering degree of polynomial kernel.')
        print ('Kernel type : polynomial')
        print ('Decision method : ovo')
    for d in range(1,5):
        score = _svm_(X, Y, kernel='poly', shape='ovo', degree=d)
        scores.append((score.mean(), score.var()))
        if print_report:
            print("Degree = {0} --> Accuracy: {1} +- {2}".format(float(d), score.mean(), score.var()))
    return scores

################################################################################

# Load the data
x, y = load_data()

# Shuffle data
X, Y = shuffle(x,y) 

## One vs one approach
default_kernel_evaluations(X, Y, shape='ovo', plot=False, title='Accuracy Scores of Different Kernels for\nOne vs One Approach with Uniform Class Weights')
## One vs all approach
default_kernel_evaluations(X, Y, shape='ovr', plot=False, title='Accuracy Scores of Different Kernels for\nOne vs All Approach with Uniform Class Weights')
## One vs one approach with class weights
default_kernel_evaluations(X, Y, shape='ovo', weights='balanced', plot=False, title='Accuracy Scores of Different Kernels for\nOne vs One Approach with Balanced Class Weights')


## Change penalty parameter C of the error term.
report = True
alter_penalty_parameter(X, Y, kernel='linear', shape='ovo', print_report=report)
alter_penalty_parameter(X, Y, kernel='poly', shape='ovo', print_report=report)
alter_penalty_parameter(X, Y, kernel='rbf', shape='ovo', print_report=report)
alter_penalty_parameter(X, Y, kernel='sigmoid', shape='ovo', print_report=report)

# Change gamma for rbf, poly and sigmoid kernels.
alter_gamma(X, Y, kernel='poly', shape='ovo', print_report=report)
alter_gamma(X, Y, kernel='rbf', shape='ovo', print_report=report)
alter_gamma(X, Y, kernel='sigmoid', shape='ovo', print_report=report)

# Change the degree for polynomial kernel.
alter_poly_degree(X, Y, print_report=report)