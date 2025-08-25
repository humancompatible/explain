import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


class BlackBox(nn.Module):
    """
    Feedforward MLP classifier used as the oracle black-box model.
    This model consists of two linear layers to map the encoded feature space to binary logits.
    """

    def __init__(self, inp_shape: int):
        """
        Initialize the BlackBox classifier.

        Args:
            inp_shape (int): Number of input features.
        """
        super(BlackBox, self).__init__()
        self.inp_shape = inp_shape
        self.hidden_dim = 10
        self.predict_net = nn.Sequential(
            nn.Linear(self.inp_shape, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for binary classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inp_shape).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 2), where each entry
                          corresponds to the unnormalized score for classes 0 and 1.
        """
        return self.predict_net(x)
