from PIL import Image
import json
import paddle.vision.transforms as transforms
import os
import paddle.io as data
import paddle
import numpy as np

identity = lambda x: x


class SimpleDataset(data.Dataset):
    def __init__(self, data_file, transform, target_transform=identity):
        super().__init__()
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset(data.Dataset):
    def __init__(self, data_file, batch_size, transform):
        super().__init__()
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = identity

    def __getitem__(self, i):
        index = self.cl_list[i.item()]
        sub_data = np.array(self.sub_meta[index])
        ri = np.random.permutation(len(sub_data))
        sf_sub_data = sub_data[ri][:self.batch_size]
        imgs = []
        targets = []
        for ssd in sf_sub_data:
            image_path = os.path.join(ssd)
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            target = paddle.to_tensor(self.target_transform(index))
            imgs.append(img)
            targets.append(target)
        imgs = paddle.stack(imgs, axis=0)
        targets = paddle.stack(targets, axis=0)
        return imgs, targets

    def __len__(self):
        return len(self.cl_list)


class EpisodicBatchSampler(data.Sampler):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield paddle.randperm(self.n_classes)[:self.n_way]