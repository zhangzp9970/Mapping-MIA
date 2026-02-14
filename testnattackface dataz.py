import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *
from tqdm import tqdm
from torchplus.models import ResNet50FE_Dim, resnet50fe
from torchplus.utils import (
    Init,
    save_image2,
)
from piq import SSIMLoss
from torchplus.datasets import DataZFolder

if __name__ == "__main__":
    batch_size = 128
    class_num = 500
    root_dir = "./log/paper5e/logZZPMAIN.testnattackface2.resnet"
    target_feature_pkl = "/path/to/feature_extractor.pkl"
    ae_inv_pkl = "/path/to/myinversion.pkl"
    ae_cls_pkl = "/path/to/mycls.pkl"
    newfe_pkl = "/path/to/newfe.pkl"
    dataset_dir = "/path/to/VGGFacesmallz/dataset"
    h = 128
    w = 128

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=False,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"test VGGFacesmallz newfe attack vae-cls7 resnet50 128pix fmse 16e dataz",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 4

    ds = DataZFolder(root=dataset_dir)

    ds_len = len(ds)

    priv_train_ds, priv_test_ds = random_split(
        ds, [ds_len * 6 // 7, ds_len - ds_len * 6 // 7]
    )

    train_ds = priv_train_ds
    test_ds = priv_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    class NewFE(nn.Module):
        def __init__(self, in_dim, out_dim) -> None:
            super(NewFE, self).__init__()
            self.fc1 = nn.Linear(in_dim, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc_mu = nn.Linear(2048, out_dim)
            self.fc_var = nn.Linear(2048, out_dim)
            self.lrelu1 = nn.LeakyReLU()
            self.lrelu2 = nn.LeakyReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.lrelu1(x)
            x = self.fc2(x)
            x = self.lrelu2(x)
            mu = self.fc_mu(x)
            var = self.fc_var(x)
            return [mu, var]

    class AEInversion(nn.Module):
        def __init__(self, latent_dim: int = 128):
            super(AEInversion, self).__init__()
            self.deconv1 = nn.ConvTranspose2d(8192, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
            self.deconv6 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(64)
            self.bn5 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()
            self.sigmod = nn.Sigmoid()
            self.fc = nn.Linear(latent_dim, 8192)

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, 8192, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.deconv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.deconv6(x)
            x = self.sigmod(x)
            return x

    class CLS(nn.Module):

        def __init__(self, latent_dim: int = 128):
            super(CLS, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, 7)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.tanh(x)
            return x

    target_feature_extractor = resnet50fe().train(False).to(output_device)
    newfe = NewFE(ResNet50FE_Dim, 8192).train(False).to(output_device)
    aeinversion = AEInversion(8192).train(False).to(output_device)
    aecls = CLS(8192).train(False).to(output_device)

    assert os.path.exists(target_feature_pkl)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(newfe_pkl)
    newfe.load_state_dict(torch.load(open(newfe_pkl, "rb"), map_location=output_device))

    assert os.path.exists(ae_inv_pkl)
    aeinversion.load_state_dict(
        torch.load(open(ae_inv_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_cls_pkl)
    aecls.load_state_dict(
        torch.load(open(ae_cls_pkl, "rb"), map_location=output_device)
    )

    with torch.no_grad():
        r = 0
        ssimloss = 0
        mseloss = 0
        accall = 0
        accall1 = 0
        accall2 = 0
        accall3 = 0
        accall4 = 0
        accall5 = 0
        accall6 = 0
        accall7 = 0
        for i, ((im, _, properties), label) in enumerate(
            tqdm(train_dl, desc=f"private")
        ):
            r += 1
            im = im.to(output_device)
            label = label.to(output_device)
            Bald = (
                torch.unsqueeze(properties["Bald"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Black_Hair = (
                torch.unsqueeze(properties["Black_Hair"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Blond_Hair = (
                torch.unsqueeze(properties["Blond_Hair"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Brown_Hair = (
                torch.unsqueeze(properties["Brown_Hair"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Gray_Hair = (
                torch.unsqueeze(properties["Gray_Hair"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Eyeglasses = (
                torch.unsqueeze(properties["Eyeglasses"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            Male = (
                torch.unsqueeze(properties["Male"], dim=1)
                .to(torch.float)
                .to(output_device)
            )
            properties = torch.cat(
                (
                    Bald,
                    Black_Hair,
                    Blond_Hair,
                    Brown_Hair,
                    Gray_Hair,
                    Eyeglasses,
                    Male,
                ),
                dim=1,
            )
            bs, c, h, w = im.shape
            im2 = resize(im, 224, antialias=False)
            target_feature = target_feature_extractor.forward(im2)
            target_mu, target_var = newfe.forward(target_feature)
            target_std = torch.exp(0.5 * target_var)
            target_eps = torch.randn_like(target_std)
            target_feature8192 = target_eps * target_std + target_mu
            rim = aeinversion.forward(target_feature8192)
            predict = aecls.forward(target_feature8192)
            predictproperties = torch.round(predict)
            acc = torch.count_nonzero(predictproperties == properties) / (
                7 * bs
            )
            acc1 = (
                torch.count_nonzero(predictproperties[:, 0] == properties[:, 0])
                / bs
            )
            acc2 = (
                torch.count_nonzero(predictproperties[:, 1] == properties[:, 1])
                / bs
            )
            acc3 = (
                torch.count_nonzero(predictproperties[:, 2] == properties[:, 2])
                / bs
            )
            acc4 = (
                torch.count_nonzero(predictproperties[:, 3] == properties[:, 3])
                / bs
            )
            acc5 = (
                torch.count_nonzero(predictproperties[:, 4] == properties[:, 4])
                / bs
            )
            acc6 = (
                torch.count_nonzero(predictproperties[:, 5] == properties[:, 5])
                / bs
            )
            acc7 = (
                torch.count_nonzero(predictproperties[:, 6] == properties[:, 6])
                / bs
            )
            ssim = SSIMLoss()(rim, im)
            mse = nn.MSELoss()(predict, properties)
            # save_excel(predictproperties, f"{log_dir}/priv.xlsx")
            # save_excel(properties, f"{log_dir}/privprop.xlsx")
            # save_image2(im.detach(), f"{log_dir}/priv/input/{i}.png", nrow=4)
            # save_image2(rim.detach(), f"{log_dir}/priv/output/{i}.png", nrow=4)
            # break
            ssimloss += ssim
            mseloss += mse
            accall += acc
            accall1 += acc1
            accall2 += acc2
            accall3 += acc3
            accall4 += acc4
            accall5 += acc5
            accall6 += acc6
            accall7 += acc7

        ssimlossavg = ssimloss / r
        mselossavg = mseloss / r
        accallavg = accall / r
        accallavg1 = accall1 / r
        accallavg2 = accall2 / r
        accallavg3 = accall3 / r
        accallavg4 = accall4 / r
        accallavg5 = accall5 / r
        accallavg6 = accall6 / r
        accallavg7 = accall7 / r
        writer.add_text("private ssim", f"{ssimlossavg}")
        writer.add_text("private mse", f"{mselossavg}")
        writer.add_text("private acc", f"{accallavg}")
        writer.add_text("private Bald", f"{accallavg1}")
        writer.add_text("private Black_Hair", f"{accallavg2}")
        writer.add_text("private Blond_Hair", f"{accallavg3}")
        writer.add_text("private Brown_Hair", f"{accallavg4}")
        writer.add_text("private Gray_Hair", f"{accallavg5}")
        writer.add_text("private Eyeglasses", f"{accallavg6}")
        writer.add_text("private Male", f"{accallavg7}")

    writer.close()
