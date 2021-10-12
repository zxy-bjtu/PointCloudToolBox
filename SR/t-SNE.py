import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import palettable
import torch


def read_shape_names():
    global shape_names
    path = "./data/shape_names.txt"
    with open(path, "r", encoding="utf-8") as f:
        shape_names = f.read().splitlines()
        f.close()
    return shape_names


def plot_t_sne(features, shape_names, y):
    """
    点云分类的特征图
    :param features:
    """
    cnames = palettable.tableau.TableauMedium_10.hex_colors
    features = features.cpu().detach().numpy()
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_t_sne = t_sne.fit_transform(features)

    x_min, x_max = X_t_sne.min(0), X_t_sne.max(0)
    X_norm = (X_t_sne - x_min) / (x_max - x_min)  # 归一化
    plt.figure("PointNet")

    for i in range(X_norm.shape[0]):
        plt.scatter(
            x=X_norm[i, 0],
            y=X_norm[i, 1],
            color=cnames[y[i]],
            s=1
        )
    plt.xticks([])
    plt.yticks([])

    plt.legend(labels=shape_names,
               fontsize=10,
               loc=3,
               bbox_to_anchor=(1.01, 0),
               borderaxespad=0)

    plt.savefig("./data/modelnet10_pointnet_tsne.png", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    shape_names = read_shape_names()
    print(shape_names)
    y = []
    np_feature = np.load("./data/modelnet10_features.npy")
    # print("feature: \n", np_feature)
    # print(np_feature.shape)
    torch_feature = torch.from_numpy(np_feature)
    # print(torch_feature)
    y = np.load("./data/y.npy")
    torch_y = torch.from_numpy(y)
    plot_t_sne(torch_feature, shape_names, torch_y)
