import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy
from torchplus.models import resnet50fe, ResNet50FE_Dim
from torchplus.datasets import DataZFolder

if __name__ == "__main__":
    batch_size = 64
    train_epoches = 16
    log_epoch = 4
    class_num = 500
    root_dir = "./log/paper5e/logZZPMAIN.resnetface"
    dataset_dir = "/path/to/VGGFacesmallz/dataset"
    h = 128
    w = 128

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"main VGGFacesmall resnet50 dataz",
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

    ds = DataZFolder(root=dataset_dir, transform=transform)

    ds_len = len(ds)

    train_ds, test_ds = random_split(ds, [ds_len * 6 // 7, ds_len - ds_len * 6 // 7])

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
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

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    class CLS(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(CLS, self).__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            x = self.fc(x)
            return x

    feature_extractor = (
        resnet50fe(weights=ResNet50_Weights.IMAGENET1K_V2).train(True).to(output_device)
    )
    cls = CLS(ResNet50FE_Dim, class_num).train(True).to(output_device)

    optimizer_fe = optim.Adam(
        feature_extractor.parameters(), lr=0.00002, betas=(0.5, 0.999), amsgrad=True
    )

    optimizer_cls = optim.Adam(
        cls.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        iters = tqdm(train_dl, desc=f"epoch {epoch_id}")
        for i, ((im, _, properties), label) in enumerate(iters):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            im = resize(im, 224, antialias=False)
            optimizer_fe.zero_grad()
            optimizer_cls.zero_grad()
            feature = feature_extractor.forward(im)
            out = cls.forward(feature)
            ce = nn.CrossEntropyLoss()(out, label)
            loss = ce
            loss.backward()
            optimizer_cls.step()
            optimizer_fe.step()

        if epoch_id % log_epoch == 0:
            train_ca = ClassificationAccuracy(class_num)
            after_softmax = F.softmax(out, dim=-1)
            predict = torch.argmax(after_softmax, dim=-1)
            train_ca.accumulate(label=label, predict=predict)
            acc_train = train_ca.get()
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("acc_training", acc_train, epoch_id)
            with open(
                os.path.join(model_dir, f"feature_extractor_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(feature_extractor.state_dict(), f)
            with open(os.path.join(model_dir, f"cls_{epoch_id}.pkl"), "wb") as f:
                torch.save(cls.state_dict(), f)

            with torch.no_grad():
                feature_extractor.eval()
                cls.eval()
                r = 0
                celoss = 0
                after_softmax_list_by_label = [[] for i in range(class_num)]
                test_ca = ClassificationAccuracy(class_num)
                for i, ((im, _, properties), label) in enumerate(
                    tqdm(train_dl, desc="testing train")
                ):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    im = resize(im, 224, antialias=False)
                    feature = feature_extractor.forward(im)
                    out = cls.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("train loss", celossavg, epoch_id)
                writer.add_scalar("acc_train", acc_test, epoch_id)

                r = 0
                celoss = 0
                after_softmax_list_by_label = [[] for i in range(class_num)]
                test_ca = ClassificationAccuracy(class_num)
                for i, ((im, _, properties), label) in enumerate(
                    tqdm(test_dl, desc="testing test")
                ):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    im = resize(im, 224, antialias=False)
                    feature = feature_extractor.forward(im)
                    out = cls.forward(feature)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("test loss", celossavg, epoch_id)
                writer.add_scalar("acc_test", acc_test, epoch_id)

    writer.close()
