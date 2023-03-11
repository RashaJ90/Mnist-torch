from src.data.make_dataset import *


def draw_mnist(m, X, y):
    samples_idx = np.random.randint(y.size, size=m)
    y_chosen = y[samples_idx]
    while np.all(y_chosen == 0) or np.all(y_chosen == 1):
        samples_idx = np.random.randint(y.size, size=m)
        y_chosen = y[samples_idx]
    return rearrange_data(X[samples_idx, :, :]), y[samples_idx]