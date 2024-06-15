"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
import torch.nn.functional as F
import numpy as np


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):

        super().__init__()

        self.save_hyperparameters(hparams)
        self.num_classes = num_classes
        self.n_channels = 3

        self.feature_extractor = models.mobilenet_v2(pretrained=True).features.to(self.device)
        set_parameter_requires_grad(self.feature_extractor, True)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(1280, 160),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(160, 80),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(80, 40),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(40, 30, kernel_size=1),
            nn.Upsample(scale_factor=1.875, mode='bilinear', align_corners=True),
            nn.Conv2d(30, 23, kernel_size=1),
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.feature_extractor(x)
        out = self.decoder(x)
        return out

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

    def training_step(self, batch, batch_idx):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        images, targets = batch
        device = self.device
        images, targets = images.to(device), targets.to(device)
        out = self.forward(images)
        out = torch.squeeze(out)
        loss = loss_func(out, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        images, targets = batch
        device = self.device
        images, targets = images.to(device), targets.to(device)
        out = self.forward(images)
        loss = loss_func(out, targets)
        _, preds = torch.max(out, 1)
        targets_mask = targets >= 0
        acc = np.mean((preds == targets)[targets_mask].cpu().numpy())
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), self.hparams["learning_rate"])
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(
            target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
