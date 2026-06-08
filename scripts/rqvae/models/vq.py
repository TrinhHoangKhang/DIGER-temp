import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class VectorQuantizer(nn.Module):

    def __init__(self, args, n_e, sk_epsilon=0.003,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.beta = args.beta
        self.dist = 'l2'
        self.kmeans_init = args.kmeans_init
        self.kmeans_iters = args.kmeans_iters
        self.use_constrained_kmeans = args.use_constrained_kmeans
        self.sk_epsilon = sk_epsilon
        self.sk_iters = args.sk_iters
        self.tau = 0.1

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not self.kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        # print(f"[VQ] init_emb called. use_constrained_kmeans={self.use_constrained_kmeans}", flush=True)
        if self.use_constrained_kmeans:
            # Use constrained k-means for balanced initialization
            try:
                # print("[VQ] Attempting to import k_means_constrained...", flush=True)
                from k_means_constrained import KMeansConstrained
                # print("[VQ] Import successful.", flush=True)
                
                # print("[VQ] Converting data to numpy...", flush=True)
                x = data.cpu().detach().numpy()
                # print("[VQ] Conversion successful.", flush=True)

                n_samples = len(x)
                
                # Set size constraints to ensure balanced distribution
                # More conservative constraints to avoid segmentation fault
                size_min = max(n_samples // (self.n_e * 3), 5)
                size_max = max(n_samples // (self.n_e - 10), size_min * 6)
                
                # Ensure constraints are valid
                if size_max * self.n_e < n_samples:
                    size_max = (n_samples // self.n_e) + 10
                
                # print(f"[VQ] Constrained K-Means: n_samples={n_samples}, n_clusters={self.n_e}, size_min={size_min}, size_max={size_max}", flush=True)
                
                clf = KMeansConstrained(
                    n_clusters=self.n_e,
                    size_min=size_min,
                    size_max=size_max,
                    max_iter=min(self.kmeans_iters, 50),  # Reduce iterations to avoid hang
                    n_init=3,  # Reduce n_init for stability
                    n_jobs=1,  # Use single thread to avoid segfault
                    verbose=0,
                    random_state=42
                )
                clf.fit(x)
                centers = torch.from_numpy(clf.cluster_centers_).to(data.device)
                # print(f"[VQ] Constrained K-Means completed successfully", flush=True)
            except ImportError:
                # print("[VQ] Warning: k-means-constrained not installed, falling back to standard k-means", flush=True)
                # print("      Install with: pip install k-means-constrained", flush=True)
                centers = kmeans(data, self.n_e, self.kmeans_iters)
            except Exception as e:
                # print(f"[VQ] Error in Constrained K-Means: {e}", flush=True)
                # print("[VQ] Falling back to standard k-means", flush=True)
                centers = kmeans(data, self.n_e, self.kmeans_iters)
        else:
            # Standard k-means initialization
            # print("[VQ] Standard K-Means initialization", flush=True)
            centers = kmeans(data, self.n_e, self.kmeans_iters)

        self.embedding.weight.data.copy_(centers)
        self.initted = True
        # print("[VQ] Initialization done", flush=True)

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, conflict=False):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

         # Calculate the distances between latent and Embedded weights
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'dot':
            d = torch.matmul(latent, self.embedding.weight.t()) / self.tau
            d = -d
        elif self.dist.lower() == 'cos':
            d = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
            d = -d
        else:
            raise NotImplementedError

        if self.sk_epsilon > 0 and (self.training or conflict):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            # print(f"[VQ] calling sinkhorn. epsilon={self.sk_epsilon}, d_min={d.min().item()}, d_max={d.max().item()}", flush=True)
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            # Check if Sinkhorn algorithm failed (returned None)
            if Q is None:
                # print(f"[VQ] Sinkhorn Algorithm failed (returned None), falling back to argmin. Try increasing sk_epsilon (current: {self.sk_epsilon}).", flush=True)
                indices = torch.argmin(d, dim=-1)
            elif torch.isnan(Q).any() or torch.isinf(Q).any():
                # print(f"[VQ] Sinkhorn Algorithm returns nan/inf values, falling back to argmin.", flush=True)
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist.lower() in ['dot', 'cos']:
            d = - torch.matmul(F.normalize(latent.detach(), dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
            loss = self.beta * F.cross_entropy(-d, indices.detach())
        else:
            raise NotImplementedError

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        d = -d
        topk_prob, topk_idx = d.topk(maxk + 1, dim=-1)

        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk-1] == topk_prob[:, maxk-1].max())

            indices = torch.where(fix, topk_idx[:, maxk-1], topk_idx[:, maxk])


        indices = indices.view(x.shape[:-1])

        return indices, fix

class EMAVectorQuantizer(nn.Module):

    def __init__(self, args, n_e, sk_epsilon=0.003,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.beta = args.beta
        self.kmeans_init = args.kmeans_init
        self.kmeans_iters = args.kmeans_iters
        self.use_constrained_kmeans = args.use_constrained_kmeans
        self.sk_epsilon = sk_epsilon
        self.sk_iters = args.sk_iters
        self.decay = args.moving_avg_decay

        embedding = torch.randn(self.n_e, self.e_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('cluster_size', torch.ones(n_e))
        if not self.kmeans_init:
            self.initted = True
        else:
            self.initted = False

    def get_codebook(self):
        return self.embedding

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        
        if self.use_constrained_kmeans:
            # Use constrained k-means for balanced initialization
            try:
                from k_means_constrained import KMeansConstrained
                
                x = data.cpu().detach().numpy()
                n_samples = len(x)
                
                # Set size constraints to ensure balanced distribution
                # More conservative constraints to avoid segmentation fault
                size_min = max(n_samples // (self.n_e * 3), 5)
                size_max = max(n_samples // (self.n_e - 10), size_min * 6)
                
                # Ensure constraints are valid
                if size_max * self.n_e < n_samples:
                    size_max = (n_samples // self.n_e) + 10
                
                # print(f"[EMA-VQ] Constrained K-Means: n_samples={n_samples}, n_clusters={self.n_e}, size_min={size_min}, size_max={size_max}")
                
                clf = KMeansConstrained(
                    n_clusters=self.n_e,
                    size_min=size_min,
                    size_max=size_max,
                    max_iter=min(self.kmeans_iters, 50),  # Reduce iterations to avoid hang
                    n_init=3,  # Reduce n_init for stability
                    n_jobs=1,  # Use single thread to avoid segfault
                    verbose=0,
                    random_state=42
                )
                clf.fit(x)
                centers = torch.from_numpy(clf.cluster_centers_).to(data.device)
                # print(f"[EMA-VQ] Constrained K-Means completed successfully")
            except ImportError:
                # print("[EMA-VQ] Warning: k-means-constrained not installed, falling back to standard k-means")
                # print("         Install with: pip install k-means-constrained")
                centers = kmeans(data, self.n_e, self.kmeans_iters)
            except Exception as e:
                # print(f"[EMA-VQ] Error in Constrained K-Means: {e}")
                # print("[EMA-VQ] Falling back to standard k-means")
                centers = kmeans(data, self.n_e, self.kmeans_iters)
        else:
            # Standard k-means initialization
            centers = kmeans(data, self.n_e, self.kmeans_iters)

        self.embedding.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def _tile(self, x):
        n, d = x.shape
        if n < self.n_e:
            n_repeats = (self.n_e + n - 1) // n
            std = 0.01 / np.sqrt(d)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def forward(self, x, conflict=False):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.t())

        if self.sk_epsilon > 0 and (self.training or conflict):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            # Check if Sinkhorn algorithm failed (returned None)
            if Q is None:
                # print(f"[EMA-VQ] Sinkhorn Algorithm failed (returned None), falling back to argmin. Try increasing sk_epsilon (current: {self.sk_epsilon}).")
                indices = torch.argmin(d, dim=-1)
            elif torch.isnan(Q).any() or torch.isinf(Q).any():
                # print(f"[EMA-VQ] Sinkhorn Algorithm returns nan/inf values, falling back to argmin.")
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = F.embedding(indices, self.embedding).view(x.shape)

        if self.training:
            embedding_onehot = F.one_hot(indices, self.n_e).type(latent.dtype)
            embedding_sum = embedding_onehot.t() @ latent
            moving_average(self.cluster_size, embedding_onehot.sum(0), self.decay)
            moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            cluster_size = laplace_smoothing(self.cluster_size, self.n_e) * n
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embedding_normalized)

            temp = self._tile(latent)
            temp = temp[torch.randperm(temp.size(0))][:self.n_e]
            usage = (self.cluster_size.view(self.n_e, 1) >= 1).float()
            self.embedding.data.mul_(usage).add_(temp * (1 - usage))

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = 0
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.t())

        d = -d
        topk_prob, topk_idx = d.topk(maxk + 1, dim=-1)

        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk - 1] == topk_prob[:, maxk - 1].max())

            indices = torch.where(fix, topk_idx[:, maxk - 1], topk_idx[:, maxk])

        indices = indices.view(x.shape[:-1])

        return indices, fix

class GumbelVectorQuantizer(nn.Module):

    def __init__(self, args, n_e):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.h_dim = args.h_dim
        self.tau = args.temperature

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.proj = nn.Linear(self.h_dim, self.n_e, bias=False)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def forward(self, x, conflict=False):
        # Flatten input
        latent = x.view(-1, self.h_dim)

        logits = self.proj(latent)

        if self.training or conflict:
            soft_onehot = F.gumbel_softmax(logits, tau=self.tau, dim=-1, hard=False)
        else:
            soft_onehot = F.softmax(logits, dim=-1)

        indices = soft_onehot.argmax(dim=-1)

        x_q = torch.matmul(soft_onehot, self.embedding.weight)

        log_logits = F.log_softmax(logits, dim=-1)
        log_uniform = torch.full_like(log_logits, -torch.log(torch.tensor(self.n_e)))
        loss = F.kl_div(log_logits, log_uniform, reduction="batchmean", log_target=True)

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):
        # Flatten input
        latent = x.view(-1, self.h_dim)

        logits = self.proj(latent)

        soft_onehot = F.softmax(logits, dim=-1)

        topk_prob, topk_idx = soft_onehot.topk(maxk + 1, dim=-1)
        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk-1] == topk_prob[:, maxk-1].max())

            indices = torch.where(fix, topk_idx[:, maxk-1], topk_idx[:, maxk])


        indices = indices.view(x.shape[:-1])

        return indices, fix

