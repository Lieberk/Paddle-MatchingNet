import paddle
import numpy as np


def one_hot(y, num_class):
    x = paddle.zeros([len(y), num_class], dtype="int64")
    y = paddle.to_tensor(y.unsqueeze(1), dtype="int64")
    updates = paddle.ones([y.shape[0], num_class], dtype="int64")
    grid_x, grid_y = paddle.meshgrid(paddle.arange(y.shape[0]), paddle.arange(y.shape[1]))
    index = paddle.stack([grid_x.flatten(), y.flatten()], axis=1)
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(updates, index=updates_index)
    return paddle.scatter_nd_add(x, index, updates)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))

    return np.mean(cl_sparsity)
