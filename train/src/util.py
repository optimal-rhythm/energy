import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

#One-hot encode the cat vars
categories = ['none', 'uniform', 'north', 'east', 'south', 'west']
reverter = lambda x: categories.index(x)
one_hot_encoder = OneHotEncoder(categories=[range(6), range(6)])
 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
 
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = (np.vectorize(reverter))(X)
        return (one_hot_encoder.fit_transform(X)).toarray()