import os
from model import UNet, R2U_Net
import torch
import Dataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils.utils import val, DiceLoss


class Trainer:
    def __init__(self, path, model_path, model_save_path, img_save_path, split, bz, numworks, max_epoch, lr=0.0005):
        self.path = path
        self.model = model_path
        self.model_save = model_save_path
        self.img_save_path = img_save_path
        self.bz = bz

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(3, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epoch, eta_min=0)
        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = DiceLoss()

        dataset = Dataset.Datasets(path)
        train_size = int(len(dataset) * 0.8)
        validate_size = int(len(dataset) * 0.2)
        train_dataset, validate_dataset = torch.utils.data.random_split(
            dataset, [train_size, validate_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=bz, drop_last=True, shuffle=True, num_workers=numworks)
        self.validate_loader = DataLoader(
            validate_dataset, batch_size=bz, drop_last=True, shuffle=True, num_workers=numworks)

        # self.loader = DataLoader(Dataset.Datasets(path), batch_size=bz, drop_last=True, shuffle=True, num_workers=4)

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(self.model))
            print("Loaded {}".format(self.model))
        else:
            print("Does not exist.")
        os.makedirs(img_save_path, exist_ok=True)

        self.model = self.model_save + '.plt'

    def train(self, max_epoch):
        for epoch in range(1, max_epoch + 1):
            for inputs, labels in tqdm(self.train_loader, desc="Epoch {}/{}".format(epoch, max_epoch),
                                       ascii=True, total=len(self.train_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # from IPython import embed; embed(); assert 0
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ori_image = inputs[0]
                pred = outputs[0]
                pred = torch.sigmoid(pred)
                y = labels[0]
                pred = pred.repeat(3, 1, 1)
                y = y.repeat(3, 1, 1)
                img = torch.stack([ori_image, pred, y], 0)

                res = val(self.validate_loader, self.net, self.loss_func,
                          self.device, self.bz, epoch, max_epoch, self.img_save_path)
                save_image(img.cpu(), os.path.join(
                    self.img_save_path, "{}.png".format(epoch)))
                self.lr_scheduler.step()

            lrs = self.optimizer.param_groups[0]['lr']
            print("\nEpoch: {}/{}, Loss_train: {}, Loss_val: {}, lr:{}".format(epoch,
                                                                               max_epoch, loss, res, lrs))
            torch.save(self.net.state_dict(), self.model)
            if epoch % 20 == 0:
                save_path = self.model_save + 'model{}.plt'.format(epoch)
                torch.save(self.net.state_dict(), save_path)
                print("model{} is saved !".format(epoch))