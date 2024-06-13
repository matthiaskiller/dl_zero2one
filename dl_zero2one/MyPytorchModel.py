import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np


class MyPytorchModel(pl.LightningModule):

    def __init__(self, hparams, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()

        # Set hyperparams
        self.save_hyperparameters(hparams)        
        self.model = None

        # Initialize your model
        activation = getattr(nn, self.hparams["activation"])()
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hparams["n_hidden"]),
            activation,
            nn.Linear(self.hparams["n_hidden"], num_classes)
        )

    def forward(self, x):
        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)
        # feed x into model!
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        # forward pass
        out = self.forward(images)
        # loss
        loss = F.cross_entropy(out, targets)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        self.log(f'{mode}_loss', loss, prog_bar=True)
        self.log(f'{mode}_accuracy', n_correct / len(targets), prog_bar=True)
        return loss, n_correct

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        return {'loss': loss, 'train_n_correct': n_correct}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def early_stopping(self, val_loss):
        return {'val_loss': val_loss}

    def prepare_data(self):
        CIFAR_ROOT = "../datasets/cifar10"
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Define your transforms (convert to tensors, normalize)
        my_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        cifar_complete_augmented = torchvision.datasets.ImageFolder(
            root=CIFAR_ROOT, transform=my_transform)
        cifar_complete_train_val = torchvision.datasets.ImageFolder(
            root=CIFAR_ROOT, transform=train_val_transform)

        N = len(cifar_complete_augmented)
        num_train, num_val = int(N*0.6), int(N*0.2)
        np.random.seed(0)
        indices = np.random.permutation(N)
        train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train + num_val], indices[num_train+num_val:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

        # Assign to use in dataloaders
        self.dataset = {"train": cifar_complete_augmented, "val": cifar_complete_train_val, "test": cifar_complete_train_val}

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], sampler=self.sampler["train"], num_workers=self.hparams["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"], sampler=self.sampler["val"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], sampler=self.sampler["test"], num_workers=self.hparams["num_workers"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"]
        )

    def getTestAcc(self, loader=None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
