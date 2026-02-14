import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *
from tqdm import tqdm
from torchplus.utils import Init, save_image2
from torchplus.models import ResNet50FE_Dim, resnet50fe
from piq import SSIMLoss
from torchplus.datasets import DataZFolder

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 48
    log_epoch = 4
    class_num = 530
    root_dir = "./log/paper5e/logZZPMAIN.nattackface2.resnet"
    target_feature_pkl = "/path/to/feature_extractor.pkl"
    ae_feature_pkl = "/path/to/feature_extractor.pkl"
    ae_inv_pkl = "/path/to/myinversion.pkl"
    ae_cls_pkl = "/path/to/mycls.pkl"
    priv_dataset_dir = "./path/to/VGGFacesmallz/dataset"
    aux_dataset_dir = "/path/to/VGGFacesmall2z/dataset"
    h = 128
    w = 128

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"VGGFacesmall2z mattack vae-cls7 resnet50 newfe 128pix fmse dataz",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir, model_dir = init.get_log_dir()
    data_workers = 4

    transform = Compose(
        [
            RandomHorizontalFlip(),
        ]
    )

    aux_ds = DataZFolder(root=aux_dataset_dir, transform=transform)

    aux_ds_len = len(aux_ds)

    aux_train_ds, aux_test_ds = random_split(
        aux_ds, [aux_ds_len * 6 // 7, aux_ds_len - aux_ds_len * 6 // 7]
    )

    aux_train_ds_len = len(aux_train_ds)

    train_ds = aux_train_ds

    train_ds_len = len(train_ds)

    print(train_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
        pin_memory=True,
    )

    train_dl_len = len(train_dl)

    class AEFeatureExtracter(nn.Module):
        def __init__(self, latent_dim: int = 8192):
            super(AEFeatureExtracter, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.mp4 = nn.MaxPool2d(2, 2)
            self.mp5 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()
            self.fc_mu = nn.Linear(8192, latent_dim)
            self.fc_var = nn.Linear(8192, latent_dim)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.mp5(x)
            x = self.relu5(x)
            x = x.view(-1, 8192)
            mu = self.fc_mu(x)
            var = self.fc_var(x)
            return [mu, var]

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
    ae_feature_extractor = AEFeatureExtracter(8192).train(False).to(output_device)
    newfe = NewFE(ResNet50FE_Dim, 8192).train(True).to(output_device)
    aeinversion = AEInversion(8192).train(False).to(output_device)
    aecls = CLS(8192).train(False).to(output_device)

    optimizer = optim.Adam(
        newfe.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    assert os.path.exists(target_feature_pkl)
    target_feature_extractor.load_state_dict(
        torch.load(open(target_feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_feature_pkl)
    ae_feature_extractor.load_state_dict(
        torch.load(open(ae_feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_inv_pkl)
    aeinversion.load_state_dict(
        torch.load(open(ae_inv_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(ae_cls_pkl)
    aecls.load_state_dict(
        torch.load(open(ae_cls_pkl, "rb"), map_location=output_device)
    )

    target_feature_extractor.requires_grad_(False)
    ae_feature_extractor.requires_grad_(False)
    aeinversion.requires_grad_(False)
    aecls.requires_grad_(False)

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, ((im, _, properties), label) in enumerate(
            tqdm(train_dl, desc=f"epoch {epoch_id} ")
        ):
            im = im.to(output_device)
            im2 = resize(im, 224, antialias=False)
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
                (Bald, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Eyeglasses, Male),
                dim=1,
            )
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            target_feature = target_feature_extractor.forward(im2)
            target_mu, target_var = newfe.forward(target_feature)
            ae_mu, ae_var = ae_feature_extractor.forward(im)
            ae_std = torch.exp(0.5 * ae_var)
            ae_eps = torch.randn_like(ae_std)
            ae_feature8192 = ae_eps * ae_std + ae_mu
            target_std = torch.exp(0.5 * target_var)
            target_eps = torch.randn_like(target_std)
            target_feature8192 = target_eps * target_std + target_mu
            fmse = F.mse_loss(target_feature8192, ae_feature8192)
            cossim = torch.mean(F.cosine_similarity(target_feature8192, ae_feature8192))
            rim = aeinversion.forward(target_feature8192)
            ssim = SSIMLoss()(rim, im)
            loss = fmse
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            predict = aecls.forward(target_feature8192)
            predictproperties = torch.round(predict)
            acc = torch.count_nonzero(predictproperties == properties) / (7 * bs)
            acc1 = torch.count_nonzero(predictproperties[:, 0] == properties[:, 0]) / bs
            acc2 = torch.count_nonzero(predictproperties[:, 1] == properties[:, 1]) / bs
            acc3 = torch.count_nonzero(predictproperties[:, 2] == properties[:, 2]) / bs
            acc4 = torch.count_nonzero(predictproperties[:, 3] == properties[:, 3]) / bs
            acc5 = torch.count_nonzero(predictproperties[:, 4] == properties[:, 4]) / bs
            acc6 = torch.count_nonzero(predictproperties[:, 5] == properties[:, 5]) / bs
            acc7 = torch.count_nonzero(predictproperties[:, 6] == properties[:, 6]) / bs
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("ssim", ssim, epoch_id)
            writer.add_scalar("cossim", cossim, epoch_id)
            writer.add_scalar("acc", acc, epoch_id)
            writer.add_scalar("acc1", acc1, epoch_id)
            writer.add_scalar("acc2", acc2, epoch_id)
            writer.add_scalar("acc3", acc3, epoch_id)
            writer.add_scalar("acc4", acc4, epoch_id)
            writer.add_scalar("acc5", acc5, epoch_id)
            writer.add_scalar("acc6", acc6, epoch_id)
            writer.add_scalar("acc7", acc7, epoch_id)
            save_image2(im.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image2(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(os.path.join(model_dir, f"newfe_{epoch_id}.pkl"), "wb") as f:
                torch.save(newfe.state_dict(), f)

    writer.close()
