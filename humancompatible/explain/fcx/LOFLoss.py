
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class LOFLoss(nn.Module):
    """
    Local Outlier Factor (LOF) as a differentiable loss for anomaly detection.

    This loss computes the average LOF score over a batch of input vectors,
    penalizing points that are outliers in the feature space.

    Args:
        n_neighbors (int): Number of nearest neighbors to consider (excluding self) when
            computing reachability distances. Default is 20.
        epsilon (float): Small constant to ensure numerical stability (no zero distances
            or division by zero). Default is 1e-8.
    """
    def __init__(self, n_neighbors=20, epsilon=1e-8):
        super(LOFLoss, self).__init__()
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    def pairwise_distances(self, x):
        """
        Compute the pairwise Euclidean distances between rows of x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: A (batch_size × batch_size) distance matrix, where entry
            (i, j) is the Euclidean distance between x[i] and x[j].
        """
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, x.t())
        dist = torch.sqrt(torch.clamp(dist, min=self.epsilon))  # Ensure no negative values before sqrt
        return dist

    def forward(self, input):
        """
        Compute the LOF loss (mean LOF score) for the input batch.

        Steps:
          1. Compute pairwise distances.
          2. Identify the k nearest neighbors for each point (excluding self).
          3. Compute reachability distances: max(distance_to_neighbor, neighbor_kth_distance).
          4. Compute local reachability density (LRD) for each point.
          5. Compute LOF score as the average ratio of neighbors' LRD to own LRD.
          6. Return the mean LOF score over all points.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Scalar tensor containing the mean LOF score across the batch.
        """
        # Ensure input is a PyTorch tensor and on the correct device
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)
        device = input.device

        # Compute pairwise distances
        distances = self.pairwise_distances(input)

        # Get the indices of the nearest neighbors (excluding self)
        knn_distances, knn_indices = torch.topk(distances, self.n_neighbors + 1, dim=1, largest=False)
        knn_distances, knn_indices = knn_distances[:, 1:], knn_indices[:, 1:]

        # Compute reachability distances
        reachability_distances = torch.max(knn_distances, distances.gather(1, knn_indices))

        # Compute local reachability density
        lrd = self.n_neighbors / (reachability_distances.sum(dim=1) + self.epsilon)  # Avoid division by zero

        # Compute LOF scores
        lof_scores = torch.zeros(input.size(0), device=device)
        for i in range(input.size(0)):
            lrd_ratios = lrd[knn_indices[i]] / (lrd[i] + self.epsilon)  # Avoid division by zero
            lof_score = lrd_ratios.sum() / self.n_neighbors
            lof_scores[i] = lof_score

        # Compute the loss as the mean of LOF scores
        loss = torch.mean(lof_scores)
        return loss
