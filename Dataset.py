import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image


class Datasets(Dataset):
    def __init__(self, path, split='train'):
        self.split = split
        self.path = path + split + '/'
        self.trans = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

        if split == 'train':
            self.images = os.listdir(os.path.join(self.path, "image/"))
            self.label = os.listdir(os.path.join(self.path, "label/"))
        else:
            self.images = os.listdir(os.path.join(self.path))

        self.len = len(self.images)

    def __len__(self):
        return len(self.images)

    def resize(self, img, size):
        h, w = img.shape[0:2]
        _w = _h = size

        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        new_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        if self.split == 'train':
            images = self.images[index]
            label = images[0:2] + '_manual1.tif'

            img_path = [os.path.join(self.path, i) for i in ("image", "label")]
            image = cv2.imread(os.path.join(img_path[0], images))
            label_image = cv2.imread(os.path.join(img_path[1], label))

            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.resize(image, 256)
            label_image = self.resize(label_image, 256)
            return self.trans(image), self.trans(label_image)
        else:
            images = self.images[index]
            label = images[0:2] + '_test.tif'
            label_image = cv2.imread(os.path.join(self.path, label))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.resize(img, 256)
            return self.trans(img)
