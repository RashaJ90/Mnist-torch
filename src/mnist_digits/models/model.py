import numpy as np
import torch.nn as nn
import torch.nn.functional as func

np.random.seed(42)


def general_score(model, X, y):
    acc = np.mean((model.predict(X) - y) == 0)
    TP = np.sum(np.logical_and(model.predict(X) == 1, y == 1))
    FN = np.sum(np.logical_and(model.predict(X) == -1, y == 1))
    FP = np.sum(np.logical_and(model.predict(X) == 1, y == -1))
    TN = np.sum(np.logical_and(model.predict(X) == -1, y == -1))
    FPTN = max(1, FP + TN)
    TPFN = max(1, TP + FN)
    TPFP = max(1, TP + FP)
    out_dict = {'num_samples': len(y),
                'error': 1 - acc,
                'accuracy': acc,
                'FPR': FP / FPTN,
                'TPR': TP / TPFN,
                'precision': TP / TPFP,
                'recall': TP / TPFN}
    return out_dict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, training=self.training)
        x = self.fc2(x)
        return func.log_softmax(x)

