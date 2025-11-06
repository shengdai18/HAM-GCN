import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs


class ChebyshevPoly(nn.Module):
    """K-order Chebyshev polynomial approximation with node masking"""

    def __init__(self, K, in_features, out_features):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, L_norm, mask=None):
        """
        x: [batch_size, N, in_features]
        L_norm: [N, N] normalized Laplacian
        mask: [N] binary mask (1 = keep, 0 = mask)
        Returns: [batch_size, N, out_features]
        """
        batch_size, N, _ = x.shape

        if mask is not None:
            # Apply mask to input
            x = x * mask.unsqueeze(0).unsqueeze(-1)

        # Initialize Chebyshev polynomials
        Tx = [x]  # T0(x) = x
        if self.K > 1:
            # T1(x) = Lx
            x1 = torch.bmm(L_norm.unsqueeze(0).expand(batch_size, -1, -1), x)
            Tx.append(x1)

            # Higher order terms Tk(x) = 2*L*Tk-1(x) - Tk-2(x)
            for k in range(2, self.K):
                x2 = 2 * torch.bmm(L_norm.unsqueeze(0).expand(batch_size, -1, -1), Tx[-1]) - Tx[-2]
                Tx.append(x2)

        # Stack along new dimension [batch, K, N, in_features]
        Tx = torch.stack(Tx, dim=1)

        # Combine with learned weights [K, in, out]
        out = torch.einsum('bkni,kio->bno', Tx, self.weight)

        if mask is not None:
            # Apply mask to output
            out = out * mask.unsqueeze(0).unsqueeze(-1)

        return out


class DGCNN_ProgressiveMask(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_classes, K=2, mask_ratios=[0.0, 0.3, 0.6]):
        """
        Args:
            mask_ratios: List of mask ratios for each layer (0.0 = no masking, 1.0 = mask all)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.mask_ratios = mask_ratios

        # Learnable adjacency matrix with symmetry constraint
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.final_adj = None  # To store the final adjacency matrix

        # Graph convolution layers with progressive masking
        self.gconv1 = ChebyshevPoly(K, input_dim, hidden_dim)
        self.gconv2 = ChebyshevPoly(K, hidden_dim, hidden_dim)
        self.gconv3 = ChebyshevPoly(K, hidden_dim, hidden_dim)

        # 1x1 convolution
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * num_nodes, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Learnable mask parameters for each layer
        self.mask_params = nn.ParameterList([
            nn.Parameter(torch.randn(num_nodes)) for _ in range(len(mask_ratios))
        ])

    def get_mask(self, mask_param, ratio):
        """
        Generate binary mask based on learned parameters and target ratio
        Args:
            mask_param: [num_nodes] learned mask parameters
            ratio: target mask ratio
        Returns:
            binary mask [num_nodes] where 1 = keep, 0 = mask
        """
        if ratio <= 0.0:
            return torch.ones(self.num_nodes, device=mask_param.device)

        # Get top-k nodes to keep based on mask_param values
        k = int(self.num_nodes * (1 - ratio))
        _, indices = torch.topk(mask_param, k)

        # Create binary mask
        mask = torch.zeros(self.num_nodes, device=mask_param.device)
        mask[indices] = 1.0

        return mask

    def forward(self, x):
        """
        x: [batch_size, num_nodes, input_dim] (EEG features per channel)
        Returns:
            output: classification logits (already log_softmax applied)
            adj_matrix: learned adjacency matrix
            masks: list of masks used at each layer
        """
        batch_size = x.size(0)

        # Dynamic adjacency matrix with ReLU and symmetry
        W = F.relu(self.adj)
        W = (W + W.t()) / 2  # Enforce symmetry

        # Store the final adjacency matrix
        self.final_adj = W.detach().cpu().numpy()

        # Degree matrix
        D = torch.diag(torch.sum(W, dim=1))

        # Laplacian matrix
        L = D - W

        # Normalized Laplacian (Eq. in paper)
        with torch.no_grad():
            # Compute largest eigenvalue
            lambda_max = eigs(L.detach().cpu().numpy(), k=1, return_eigenvectors=False).real[0]

        L_norm = (2 / lambda_max) * L - torch.eye(self.num_nodes, device=x.device)

        # Generate masks for each layer
        masks = []
        for i, ratio in enumerate(self.mask_ratios):
            mask = self.get_mask(self.mask_params[i], ratio)
            masks.append(mask)

        # Graph convolution with progressive masking
        x = F.relu(self.gconv1(x, L_norm, masks[0]))  # [batch, N, hidden]
        x = F.relu(self.gconv2(x, L_norm, masks[1]))
        x = F.relu(self.gconv3(x, L_norm, masks[2]))

        # 1x1 convolution along feature dimension
        x = x.permute(0, 2, 1)  # [batch, hidden, N]
        x = F.relu(self.conv1x1(x))

        # Flatten and classify
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply log_softmax to the output before returning
        return F.log_softmax(x, dim=1)

    def get_final_adjacency(self):
        """Returns the final learned adjacency matrix"""
        return self.final_adj