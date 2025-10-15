from functools import cached_property
from math import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz
from sklearn.manifold import TSNE
from torch import Tensor as T_

from autograd.src import PyTensor


class MyDataset:
    """Custom dataset"""

    def __init__(self, x: T_, y: T_):
        self.x = x.float()
        self.y = y.long()

        onehot = torch.zeros(self.num_samples, self.num_classes)
        for i in range(onehot.shape[0]):
            for j in range(onehot.shape[1]):
                if j == self.y[i]:
                    onehot[i, j] = 1
        self.y_onehot = onehot.float()

    @cached_property
    def num_samples(self) -> int:
        return len(self.x)
    
    @cached_property
    def num_classes(self) -> int:
        return int(torch.max(self.y).item() + 1)


def generate_toy_data(
    feature_dim: int,
    sample_num: int,
    seed: int = 3
) -> tuple[MyDataset, MyDataset]:
    """Generating toy-samples"""

    torch.manual_seed(seed)
    feature_dim = 4
    sample_num = 1000
    data_tensor = torch.rand((sample_num, feature_dim))

    coeffs = torch.randn(2, 4)  # AUXILLIARY FUNCTION FOR ASSIGNING DATA CLASS

    def f(x, y, z, w):        
        val1 = torch.dot(
            coeffs[0],
            torch.tensor([sin(x), cos(y), asin(z), acos(w)]),
        )
        val2 = torch.dot(
            coeffs[1],
            torch.tensor([exp(x), exp(y), cosh(z), sinh(w)]),
        )
        return val1, val2

    y1, y2 = torch.zeros(sample_num), torch.zeros(sample_num) 

    for i in range(sample_num):
        y1[i], y2[i] = f(*data_tensor[i])

    # Generating data classes

    # fact1 = y1 < -0.5
    # fact2 = y2 < 0.6
    fact1 = y1 < 1.6
    fact2 = y2 < 2.7

    data_class = torch.zeros(sample_num, dtype=torch.uint8)  

    for i in range(0, sample_num):
        if (fact1[i] == True) and (fact2[i] == True):
            data_class[i] = 0
        elif (fact1[i] == True) and (fact2[i] == False):
            data_class[i] = 1
        elif (fact1[i] == False) and (fact2[i] == True):
            data_class[i] = 2
        else:
            data_class[i] = 3

    print(f"Class 0: {sum(data_class == 0)} samples")
    print(f"Class 1: {sum(data_class == 1)} samples")
    print(f"Class 2: {sum(data_class == 2)} samples")
    print(f"Class 3: {sum(data_class == 3)} samples")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = torch.arange(1, sample_num + 1)

    # 색상 팔레트 (4개 클래스)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 파랑, 주황, 초록, 빨강
    labels = ["Class 0", "Class 1", "Class 2", "Class 3"]

    for c in range(4):
        mask = data_class == c
        axes[0].scatter(x[mask], y1[mask], s=4, color=colors[c], label=labels[c], alpha=0.7)
        axes[1].scatter(x[mask], y2[mask], s=4, color=colors[c], label=labels[c], alpha=0.7)

    axes[0].set_title(r"$f_1(\mathbf{x}_i)$", fontsize=12)
    axes[1].set_title(r"$f_2(\mathbf{x}_i)$", fontsize=12)
    axes[0].set_xlabel("Data index $i$")
    axes[1].set_xlabel("Data index $i$")
    axes[0].legend(fontsize=9, loc="best")
    axes[1].legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()

    # train / test split
    randvec = torch.rand(sample_num)    
    train_mask = randvec <= 0.7
    test_mask = randvec > 0.7
    assert sum(train_mask) + sum(test_mask) == sample_num

    print(f"\nTraining set: {sum(train_mask)} samples")
    print(f"Testing set: {sum(test_mask)} samples")

    # Wrapping into custom data Class
    data_train = MyDataset(data_tensor[train_mask], data_class[train_mask])
    data_test = MyDataset(data_tensor[test_mask], data_class[test_mask])

    return data_train, data_test


class DataLoader:
    """Custom dataloader"""

    def __init__(self, data: MyDataset, bsz: int, backend: str):
        self.data = data
        self.bsz = bsz
        self._indexes = np.arange(self.data.num_samples)
        self._shuffler = np.random.default_rng()
        self.backend = backend

    def shuffle(self) -> None:
        self._shuffler.shuffle(self._indexes)
        self.data.x = self.data.x[self._indexes]
        self.data.y = self.data.y[self._indexes]
        self.data.y_onehot = self.data.y_onehot[self._indexes]

    def __iter__(self):
        for bi in range(0, self.data.num_samples, self.bsz):
            ei = min(bi + self.bsz, self.data.num_samples)
            x = self.data.x[bi: ei]
            y = self.data.y[bi: ei]
            y_onehot = self.data.y_onehot[bi: ei]
            if self.backend == "c":
                yield (
                    PyTensor(x.numpy()).to_c(),
                    PyTensor(y.numpy()).to_c(),
                    PyTensor(y_onehot.numpy()).to_c(),
                )
            elif self.backend == "torch":
                yield (x, y, y_onehot)
            else:  # self.backend == "numpy"
                yield (
                    PyTensor(x.numpy()),
                    PyTensor(y.numpy()),
                    PyTensor(y_onehot.numpy())
                )


def draw_2d_tsne(last_hiddens: np.ndarray, labels: np.ndarray) -> None:
    z = TSNE(n_components=2).fit_transform(last_hiddens)
    fig, ax = plt.subplots()
    ax: plt.Axes
    fig.set_size_inches(5, 5)
    ax.set_title('2D projection of model output',fontsize=12)
    ax.scatter(z[:, 0], z[:, 1], s=40, c=labels, cmap="Set2")
    plt.show()
    plt.close()


def visualize_results(
    train_loss: list[float],
    test_loss: list[float],
    test_acc: list[float],
    last_hiddens: T_,
    labels: T_,
) -> None:
    """For visualizing training results"""

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 3)
    fig.suptitle("Training Result", fontsize = 12)
    axes: list[plt.Axes]

    loss1, = axes[0].plot(
        range(len(train_loss)), train_loss, color='b', linewidth=2 , label="Train set"
    )
    loss2, = axes[0].plot(
        range(len(test_loss)), test_loss, color='r', linewidth=2, label="Test set"
    )
    axes[0].legend(handles = [loss1, loss2])
    axes[0].set_xlabel("Epochs")
    axes[0].set_title("Loss")
    axes[0].grid()

    acc2, = axes[1].plot(
        range(len(test_acc)), test_acc, color='r', linewidth=2, label="Test set"
    )
    axes[1].legend(handles = [acc2])
    axes[1].set_xlabel("Epochs")
    axes[1].set_title("Accuracy")
    axes[1].grid()
    plt.show()
    plt.close()

    draw_2d_tsne(last_hiddens, labels)


def draw_computational_graph(
    loss: T_, params: dict[str, T_], save_path: str
) -> None:
    torchviz.make_dot(loss, params=params).render(save_path, format="png")

    graph = plt.imread(f"{save_path}-annotated.png")  # 원본 계산 그래프에 직접 메모한 별도의 파일
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(graph)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_size_inches(6, 11)
    fig.tight_layout()
    plt.show()
    plt.close()
