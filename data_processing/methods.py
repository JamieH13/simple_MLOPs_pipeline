import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NameDropper(BaseEstimator, TransformerMixin):
    """
    Drops columns from dataframe
    Args:
        cols: list of columns to drop
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.drop(columns=self.cols, errors='ignore')


class DataLabeler(BaseEstimator, TransformerMixin):
    """
    Generates binary labels for training, according to the debt definition.
    """

    def __init__(self, debt_def=None):
        if debt_def is None:
            debt_def = {'C': 'Good_Debt', 'X': 'Good_Debt', '0': 'Good_Debt',
                        '1': 'Neutral_Debt', '2': 'Neutral_Debt',
                        '3': 'Bad_Debt', '4': 'Bad_Debt', '5': 'Bad_Debt'}
        self.debt_def = debt_def

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.replace({'STATUS': self.debt_def})

        debt_counts = data.value_counts(subset=['ID', 'STATUS']).unstack(fill_value=0)

        debt_counts.loc[(debt_counts['Good_Debt'] > debt_counts['Neutral_Debt']), 'label'] = 1
        debt_counts.loc[(debt_counts['Good_Debt'] > debt_counts['Bad_Debt']), 'label'] = 1
        debt_counts.loc[(debt_counts['Neutral_Debt'] > debt_counts['Good_Debt']), 'label'] = 0
        debt_counts.loc[(debt_counts['Neutral_Debt'] > debt_counts['Bad_Debt']), 'label'] = 1
        debt_counts.loc[(debt_counts['Bad_Debt'] > debt_counts['Good_Debt']), 'label'] = 0
        debt_counts.loc[(debt_counts['Bad_Debt'] > debt_counts['Neutral_Debt']), 'label'] = 0

        return debt_counts.reset_index()


class OneHotEncode(BaseEstimator, TransformerMixin):
    """
    One hot encodes columns
    Args:
        cols: list of columns to encode
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        one_hot = pd.get_dummies(data[self.cols])

        data = data.join(one_hot)

        data = data.drop(columns=self.cols)
        return data


class LabelJoiner(BaseEstimator, TransformerMixin):
    """
    Joins binary labels onto dataset using overlapping ID column
    Args:
        labels: dataframe of ID and labels
    """

    def __init__(self, labels):
        self.labels = labels

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.merge(self.labels, on='ID', how='left')
        return data


class ColumnMapper(BaseEstimator, TransformerMixin):
    """
    Replace values in a column with others
    Args:
        col: column to replace values in
        mapper: dictionary of mapping
    """

    def __init__(self, col, mapper):
        self.mapper = mapper
        self.col = col

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data[self.col] = data[self.col].replace(self.mapper)
        return data


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scale all values in a column to between 0 and 1
    Args:
        col: column to scale
    """

    def __init__(self, col):
        self.col = col

    def fit(self, data, y=None):
        training = data[data.train == 1]

        self.max_value = training[self.col].max()
        self.min_value = training[self.col].min()
        return self

    def transform(self, data):
        data[self.col] = (data[self.col] - self.min_value) / (self.max_value - self.min_value)
        return data


class DropDuplicates(BaseEstimator, TransformerMixin):
    """
    Drop duplicates entries in table
    Args:
        col: list of columns to drop duplicates based on
    """

    def __init__(self, col):
        self.col = col

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.drop_duplicates(subset=[self.col])
        return data


class TrainTestSplit(BaseEstimator, TransformerMixin):
    """
    Generate a 'train' column which flags whether an entry is train, test or unlabelled
    Args:
        training_size: fraction of entries that are in the traing set
    """

    def __init__(self, training_size=0.7):
        self.training_size = training_size

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data = data.sort_values('label', ascending=True)

        num_labelled = data.label.notnull().sum()
        num_unlabelled = data.label.isna().sum()

        training = np.random.choice(a=[1, 0], size=num_labelled, p=[self.training_size, 1 - self.training_size])

        training = np.pad(training, (0, num_unlabelled), constant_values=-1)

        data['train'] = training
        return data
















