import numpy as np
from sklearn.preprocessing import LabelEncoder


class CustomerEncoder(LabelEncoder):
    """Encode last column."""

    def fit_transform(self, y, *args, **kwargs):
        return np.concatenate([y[:, :-1], np.array(super().fit_transform(y[:, -1])).reshape(-1, 1)], axis=1)

    def transform(self, y, *args, **kwargs):
        return np.concatenate([y[:, :-1], np.array(super().transform(y[:, -1])).reshape(-1, 1)], axis=1)
