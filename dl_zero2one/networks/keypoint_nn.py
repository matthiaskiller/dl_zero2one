"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


def double_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
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
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        if self.hparams["activation"] == "ReLU":
            activation = nn.ReLU()
        elif self.hparams["activation"] == "LeakyReLU":
            activation = nn.LeakyReLU()
        else:
            activation = nn.ReLU()

        self.model = nn.Sequential(
            double_conv(self.hparams["input_channels"], self.hparams["num_channels_0"],
                        kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
            nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
                         stride=self.hparams["maxpool_stride"]),

            double_conv(self.hparams["num_channels_0"], self.hparams["num_channels_1"],
                        kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
            nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
                         stride=self.hparams["maxpool_stride"]),

            double_conv(self.hparams["num_channels_1"], self.hparams["num_channels_2"],
                        kernel_size=self.hparams["conv_kernel_size"], stride=self.hparams["conv_stride"], padding=self.hparams["padding"]),
            nn.MaxPool2d(kernel_size=self.hparams["maxpool_kernel_size"],
                         stride=self.hparams["maxpool_stride"]),


            nn.Linear(self.hparams["num_fc_0"],
                      self.hparams["num_fc_1"]),
            activation,
            nn.Linear(self.hparams["num_fc_1"],
                      self.hparams["num_fc_2"]),
            activation,
            nn.Linear(self.hparams["num_fc_2"], self.hparams["output_size"]),
            nn.Tanh,
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


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
