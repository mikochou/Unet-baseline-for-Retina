import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
from model import UNet

from torch.utils.data import DataLoader
import Dataset


def test(data_dir, model_dir, device, bz, img_save_path):
    net = UNet(3, 1).to(device)
    if os.path.exists(model_dir):
        net.load_state_dict(torch.load(model_dir))
        print("Loaded{}".format(model_dir))
    else:
        print("Does not exist.")
    os.makedirs(img_save_path, exist_ok=True)

    net.eval()

    dataset = Dataset.Datasets(data_dir, split='test')
    loader = DataLoader(dataset, batch_size=bz,
                        drop_last=True, shuffle=True, num_workers=0)
    loader = tqdm(loader)
    flag = 1
    with torch.no_grad():
        for inputs in enumerate(loader, 0):
            inputs = inputs[-1].to(device)
            outputs = net(inputs)
            ori = inputs[0]
            pred = outputs[0]
            pred = torch.sigmoid(pred)
            pred = pred.repeat(3, 1, 1)
            img = torch.stack([ori, pred], 0)
            save_image(img.cpu(), os.path.join(
                img_save_path, "{}.png".format(flag)))
            flag = flag + 1

    print("done!")
    return


if __name__ == '__main__':
    data_dir = './data/'
    model_dir = './models/model280.plt'
    device = "cpu"
    img_save_path = "./test"
    bz = 4

    test(data_dir, model_dir, device, bz, img_save_path)
