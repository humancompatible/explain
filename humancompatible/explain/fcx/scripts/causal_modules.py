import torch
import pandas as pd
import numpy as np
import networkx as nx


def causal_regularization_enhanced(outputs, adj_matrix, lambda_nc=1e-3, lambda_c=1e-3, theta=0.2):
    """
    Enhanced causal regularization to enforce dependencies in outputs based on input adjacency matrix.
    
    Args:
        outputs (torch.Tensor): Reconstructed outputs, shape (batch_size, input_dim)
        adj_matrix (torch.Tensor): Adjacency matrix from input space, shape (input_dim, input_dim)
        lambda_nc (float): Regularization strength for non-connected pairs
        lambda_c (float): Regularization strength for connected pairs
        theta (float): Covariance threshold for connected pairs
    
    Returns:
        torch.Tensor: Enhanced regularization loss
    """
    batch_size, input_dim = outputs.size()

    # Center the outputs
    outputs_centered = outputs - torch.mean(outputs, dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.mm(outputs_centered.t(), outputs_centered) / (batch_size - 1)  # Shape: (input_dim, input_dim)

    # Create masks for connected and non-connected pairs
    connected_mask = adj_matrix > 0  # Boolean mask for connected pairs
    non_connected_mask = (adj_matrix == 0) & (~torch.eye(input_dim, dtype=torch.bool, device=adj_matrix.device))

    # Regularize non-connected pairs to have zero covariance
    reg_non_connected = torch.sum(cov[non_connected_mask] ** 2)

    # Encourage connected pairs to have covariance >= theta
    # Compute the difference (theta - Cov(Y_i, Y_j)) where Cov(Y_i, Y_j) < theta
    cov_connected = cov[connected_mask]
    # Compute how much each connected pair's covariance falls below theta
    cov_diff = theta - cov_connected
    # Only consider positive differences (where covariance is below theta)
    cov_diff = torch.clamp(cov_diff, min=0)
    # Penalize the squared differences
    reg_connected = torch.sum(cov_diff ** 2)

    # Total regularization loss
    reg_loss = lambda_nc * reg_non_connected + lambda_c * reg_connected
    #print("reg_loss: {}  ----- noncon: {}  ------ connect {}".format(reg_loss, reg_non_connected, reg_connected))

    return reg_loss



def binarize_adj_matrix(adj_matrix, threshold=0.5):
    """
    Converts the adjacency matrix to binary by applying a threshold.
    
    Args:
        adj_matrix (np.ndarray): Original adjacency matrix.
        threshold (float): Threshold to determine edge existence.
    
    Returns:
        np.ndarray: Binarized adjacency matrix.
    """
    binarized = (adj_matrix > threshold).astype(int)
    return binarized

def ensure_dag(adj_matrix):
    """
    Enforce that a given adjacency matrix represents a Directed Acyclic Graph (DAG).

    Args:
        adj_matrix (np.ndarray):
            Square binary adjacency matrix of shape (n, n), where
            entry (i, j) == 1 indicates a directed edge i → j.

    Returns:
        np.ndarray:
            A modified adjacency matrix of the same shape, guaranteed to be acyclic (a DAG).
    """
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph())
    
    try:
        cycle = nx.find_cycle(G, orientation='original')
        print("Cycle detected. Attempting to remove cycles.")
        # Remove edges until no cycles remain
        while True:
            try:
                cycle = nx.find_cycle(G, orientation='original')
                # Remove the last edge in the cycle
                edge_to_remove = cycle[-1][0], cycle[-1][1]
                G.remove_edge(*edge_to_remove)
                print(f"Removed edge: {edge_to_remove}")
            except nx.NetworkXNoCycle:
                print("No more cycles detected.")
                break
    except nx.NetworkXNoCycle:
        print("No cycles detected. Adjacency matrix is a DAG.")
    
    return nx.to_numpy_matrix(G).astype(int)