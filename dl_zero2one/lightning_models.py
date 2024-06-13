
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class TwoLayerNet(pl.LightningModule):
    def __init__(self, hparams, input_size=1 * 28 * 28, hidden_size=512, num_classes=10):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
        )
        self.validation_step_outputs_loss = []
        self.validation_step_outputs_acc = []



    def forward(self, x):
        # flatten the image  before sending as input to the model
        N, _, _, _ = x.shape
        x = x.view(N, -1)

        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Log the accuracy and loss values to the tensorboard
        self.log('loss', loss)
        self.log('acc', acc)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)
        
        self.validation_step_outputs_loss.append(loss)
        self.validation_step_outputs_acc.append(acc)


        # Visualise the predictions  of the model
        if batch_idx == 0:
            self.visualize_predictions(images.detach(), out.detach(), targets)

        return {'val_loss': loss, 'val_acc': acc}
        
    def on_validation_epoch_end(self):
        
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack(self.validation_step_outputs_loss).mean()
        avg_acc = torch.stack(self.validation_step_outputs_acc).mean()
        
        # Log the validation accuracy and loss values to the tensorboard
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.validation_step_outputs_loss.clear()  # free memory
        self.validation_step_outputs_acc.clear()  # free memory

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(
        ), self.hparams["learning_rate"], momentum=0.9)

        return optim

    def visualize_predictions(self, images, preds, targets):

        # Helper function to help us visualize the predictions of the
        # validation data by the model

        class_names = ['t-shirts', 'trouser', 'pullover', 'dress',
                       'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        # determine size of the grid based for the given batch size
        num_rows = torch.tensor(len(images)).float().sqrt().floor()

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(num_rows, len(images) // num_rows + 1, i+1)
            plt.imshow(images[i].cpu().numpy().squeeze(0))
            plt.title(class_names[torch.argmax(preds, axis=-1)
                                  [i]] + f'\n[{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure(
            'predictions', fig, global_step=self.global_step)


    def visualize_predictions(self, images, preds, targets):
        class_names = ['t-shirts', 'trouser', 'pullover', 'dress',
                       'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        num_rows = int(torch.sqrt(torch.tensor(images.size(0))))  # Convert tensor to int
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(num_rows, len(images) // num_rows + 1, i+1)
            plt.imshow(images[i].cpu().numpy().squeeze(0))
            plt.title(class_names[torch.argmax(preds, axis=-1)
                                  [i]] + f'[{class_names[targets[i]]}]')
        plt.show()
