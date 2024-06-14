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


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        if self.hparams["activation"] == "ReLU":
            activation = nn.ReLU()
        elif self.hparams["activation"] == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Tanh()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hparams["n_hidden_0"]),
            activation,
            nn.Linear(self.hparams["n_hidden_0"], self.hparams["n_hidden_1"]),
            activation,
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_2"]),
            activation,
            nn.Linear(self.hparams["n_hidden_2"], self.hparams["latent_dim"]),
        )

    def forward(self, x):
        return self.encoder(x)
    
    def __len__(self):
        return len(self.encoder)
    
    def __getitem__(self, key):
        return self.encoder[key]


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None
        self.output_size = output_size

        if self.hparams["activation"] == "ReLU":
            activation = nn.ReLU()
        elif self.hparams["activation"] == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Tanh()

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams["latent_dim"], self.hparams["n_hidden_2"]),
            activation,
            nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_1"]),
            activation,
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_0"]),
            activation,
            nn.Linear(self.hparams["n_hidden_0"], self.output_size),
        )

    def forward(self, x):
        return self.decoder(x)
    
    def __len__(self):
        return len(self.decoder)
    
    def __getitem__(self, key):
        return self.decoder[key]


class Autoencoder(pl.LightningModule):

    def __init__(self, hparams, encoder, decoder, train_set, val_set, logger):
        super().__init__()
        self.save_hyperparameters(hparams)
        # set hyperparams
        self.encoder = encoder
        self.decoder = decoder
        self.train_set = train_set
        self.val_set = val_set
        self.save_loggers= logger
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def general_step(self, batch, batch_idx, mode):
        images = batch
        flattened_images = images.view(images.shape[0], -1)
        reconstruction = self.forward(flattened_images)
        loss = F.mse_loss(reconstruction, flattened_images)
        return loss, reconstruction

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "train")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        flattened_images = images.view(images.shape[0], -1)
        reconstruction = self.forward(flattened_images)
        loss = F.mse_loss(reconstruction, flattened_images)

        reconstruction = reconstruction.view(reconstruction.shape[0], 28, 28).cpu().numpy()
        images = np.zeros((len(reconstruction), 3, 28, 28))
        for i in range(len(reconstruction)):
            images[i, 0] = reconstruction[i]
            images[i, 2] = reconstruction[i]
            images[i, 1] = reconstruction[i]
        self.logger.experiment.add_images('reconstructions', images, self.current_epoch, dataformats='NCHW')

        self.log('val_loss', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            [{'params': self.encoder.parameters(), 'params': self.decoder.parameters()}], 
            lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"]
        )
        return optim

    def getReconstructions(self, loader=None):
        self.eval()
        self.to(self.device)

        if not loader:
            loader = self.val_dataloader()

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(pl.LightningModule):

    def __init__(self, hparams, encoder, train_set=None, val_set=None, test_set=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = encoder
        self.data = {'train': train_set, 'val': val_set, 'test': test_set}
        self.num_classes = 10

        if self.hparams["activation"] == "ReLU":
            activation = nn.ReLU()
        elif self.hparams["activation"] == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Tanh()

        self.model = nn.Sequential(
            nn.Linear(self.hparams["latent_dim"], self.hparams["n_hidden_class_0"]),
            nn.BatchNorm1d(self.hparams["n_hidden_class_0"]),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(self.hparams["n_hidden_class_0"], self.hparams["n_hidden_class_1"]),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(self.hparams["n_hidden_class_1"], self.num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        flattened_images = images.view(images.shape[0], -1)
        out = self.forward(flattened_images)
        loss = F.cross_entropy(out, targets)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        self.log('train_loss', loss)
        self.log('train_n_correct', n_correct)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        self.log('val_loss', loss)
        self.log('val_n_correct', n_correct)
        return loss

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        self.log('test_loss', loss)
        self.log('test_n_correct', n_correct)
        return loss

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss', avg_loss)
        self.log('val_acc', acc)
        return avg_loss

    def train_dataloader(self):
        return DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return DataLoader(self.data['test'], batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
        return optim

    def getAcc(self, loader=None):
        self.eval()
        self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

    def __len__(self):
        return len(self.model)
    
    def __getitem__(self, key):
        return self.model[key]
