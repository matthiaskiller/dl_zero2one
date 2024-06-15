"""Models for facial keypoint detection"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


def double_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def double_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        
        if self.hparams["activation"] == "ReLU":
            activation = nn.ReLU()
        elif self.hparams["activation"] == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.ReLU()

        # self.conv_layers = nn.Sequential(
        #     double_conv(self.hparams["input_channels"], self.hparams["num_channels_0"],
        #                 kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
        #     nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
        #                  stride=self.hparams["maxpool_stride"]),

        #     double_conv(self.hparams["num_channels_0"], self.hparams["num_channels_1"],
        #                 kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
        #     nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
        #                  stride=self.hparams["maxpool_stride"]),

        #     double_conv(self.hparams["num_channels_1"], self.hparams["num_channels_2"],
        #                 kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
        #     nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
        #                  stride=self.hparams["maxpool_stride"]),
        # )

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(self.hparams["num_fc_0"], self.hparams["num_fc_1"]),
        #     activation,
        #     nn.Linear(self.hparams["num_fc_1"], self.hparams["num_fc_2"]),
        #     activation,
        #     nn.Linear(self.hparams["num_fc_2"], self.hparams["output_size"]),
        #     nn.Tanh(),
        # )



        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),


            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),


            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(1, -1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Flatten(1, -1),
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),


            nn.Linear(100, 30),
            nn.Tanh(),
        )

        
        self.validation_step_outputs = []

    def forward(self, x):
        # x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = self.fc_layers(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["keypoints"]
        out = self.forward(images)
        f_loss = nn.MSELoss()
        loss = f_loss(out, torch.squeeze(targets).view(out.shape))
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["keypoints"]
        out = self.forward(images)
        f_loss = nn.MSELoss()
        loss = f_loss(out, torch.squeeze(targets).view(out.shape))
        acc = 1.0 / (2 * (loss / 298))
        if batch_idx == 1:
            self.visualize_predictions(images.detach(), out.detach(), targets)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', avg_loss)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        return optim

    def visualize_predictions(self, images, preds, targets):
        num_rows = torch.tensor(len(images)).float().sqrt().floor()

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            keypoints = targets[i]
            pred_kpts = preds[i]
            plt.subplot(int(num_rows), int(len(images) // num_rows + 1), i + 1)
            plt.axis('off')
            image = (images[i].cpu().clone() * 255).view(96, 96)
            plt.imshow(image, cmap='gray')
            keypoints = keypoints.cpu().clone() * 48 + 48
            plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker='.', c='m')
            if pred_kpts is not None:
                pred_kpts = pred_kpts.cpu().clone().view(-1, 15, 2) * 48 + 48
                plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], s=200, marker='.', c='r')

        self.logger.experiment.add_figure('predictions', fig, global_step=self.global_step)

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""

    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
