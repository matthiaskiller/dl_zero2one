"""Definition of all datasets"""

from .base_networks import DummyNetwork
from .classification_net import ClassificationNet, MyOwnNetwork
from .loss import L1, MSE, BCE, CrossEntropyFromLogits
