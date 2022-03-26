"""
This code below is an implementation of Support Vector Machine (SVM)
SVM can be used for both classification (mostly) or regression

--------------
SVM algorithm:
a hyperplane is a plane that diffrentiate two classes in SVM
each data item is ploted as a point in an n-dimensional space (where n is the number of features)
the value of each features is the value of a particular coordinates
and then perform classification by finding the hyperplane that diffrentiates the two classes i.e maximize the margin between the hyperplane and the data points in n-dimensional space
the closest points on both sides of the margin (plane) are called support vectors
SVM only consider the support vectors when creating a hyperplane

For Hard margin SVM, they must be linearly seprable
For Soft margin SVM (here the datapoints are inseprable), it does consider outliers to some points

for datapoint that are non linearly seprable, SVM transfers the datapoint to a higher dimesion by adding new dimensions until the data is linearly seprable or close too (i.e new features to the data)
SVM selects the hyperplane that clasifies the classes prior to maximizing margin

Kernel transformation is the process of using existing features to generate new features. where the new feature is the kernel
Types of kernel transformation in SVM: 
- Gaussian kernel or RBF Radial basis function for non linearly data (most used)
- Linear Kernel
- Polynomial Kernel (Computationally complex)

Adavantages of SVM:
SVM Classifier is best to segregate two classes. on smaller datasets, but on complex ones.
effective in high dimensional spaces
effective where the number of dimensions is greate than the number of samples
it uses a subset of training points in the decision functions

NOTE: SVM doesn't work well for data that are not linearly seprarble (i.e with outliers)
------------------

Structure of code:
==  Setup and initialize weight and biases
==  Map class labels from {0,1} to {-1,1}
==  Perform gradient descent

"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Initialize parameters and defining helper functions to initialize weights and biases

class SVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)  # Initialize weights
        self.b = 0  # Initialize bias

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)  # if y=0 then map to -1 else 1

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1

    def _get_gradients(self, constrain, x, idx):

        # hyperplane lies on the good side of the margin
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db

        # hyperplane lies on the wrong side of the margin
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db

    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)  # check if datapoint satisfy constraint
                dw, db = self._get_gradients(constrain, x, idx)  # compute gradient
                self._update_weights_bias(dw, db)  # update weights and biases

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)

################
## Testing our algorithm

X, y = make_blobs(
    n_samples=500, n_features=3, centers=2, cluster_std=1.05, random_state=22
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
clf = SVM(n_iters=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

acc = accuracy(y_test, pred)
print (acc)