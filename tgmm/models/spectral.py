import torch
import torch.nn.functional as F
import tensorly as tl

from ..utils import logger
from .base import _BatchFitMixin


tl.set_backend("pytorch")


class GaussianMixtureSpectral(_BatchFitMixin):
    r"""A python implementation of the algorithm in the paper

    **Learning mixtures of spherical Gaussians: moment methods and spectral decompositions**
    by Hsu and Kakade (Theorem 1 therein)

    with some minor modifications according to
    **Tensor Decompositions for Learning Latent Variable Models**
    By Anandkumar, Ge, Hsu, Kakade and Telgarsky

    The impl is partially inspired by the Matlab implementation in
    https://github.com/avikdelta/Tensor_decomposition/tree/master/GMM
    """

    def __init__(self, n_components, normalize_weights=True, verbose=False, **kwargs):
        self.n_components = n_components
        self.normalize_weights = normalize_weights
        self.tensor_factorizer = tl.decomposition.SymmetricCP(
            rank=self.n_components, **kwargs
        )
        self.verbose = verbose

    def fit(self, X, *args, **kwargs):
        r"""The fitting procedure

        Args:
            X(torch.Tensor): the input data of shape [n_samples, n_features]

        Returns:
            w(torch.Tensor): the mixture probabilities of components
            mu_hat(torch.Tensor): the mean of each component
        """
        k = self.n_components
        N, d = X.size()
        if d < self.n_components and self.verbose:
            logger.warn(
                "The dimension of X is smaller than the number of components, "
                "Which may cause solutions to be unreliable"
            )
        mu = X.mean(dim=0, keepdim=True)  # [1, d]
        xox = X.T @ X / N
        cov = xox - mu.T @ mu  # [d, d]
        S1, V1 = torch.linalg.eigh(cov)
        sigma_sqr_est = S1[0]
        v = V1[:, 0].view(-1, 1)  # [d, 1]
        coeff = (X - mu) @ v  # [n, 1]
        M1 = (X * (coeff**2)).mean(dim=0)  # [d]
        M2 = xox - sigma_sqr_est * torch.eye(d)
        S2, V2 = torch.linalg.eigh(M2)
        V = V2[:, -k:]  # [d, k]
        S = S2[-k:]
        D = torch.diag(torch.sqrt(S))
        Dinv = torch.diag(1 / torch.sqrt(S))
        # Whitening matrix
        W = V @ Dinv  # [d, k]
        B = V @ D  # [d, k]

        M1w = W.T @ M1.view(-1, 1)
        Xw = X @ W  # [n, k]
        T = torch.einsum("in,jn,kn->ijk", Xw.T, Xw.T, Xw.T) / N
        E = (torch.eye(d) @ W).T  # [k, d]
        M1wr = M1w.repeat(1, d)
        T -= torch.einsum("in,jn,kn->ijk", M1wr, E, E)
        T -= torch.einsum("in,jn,kn->ijk", E, M1wr, E)
        T -= torch.einsum("in,jn,kn->ijk", E, E, M1wr)
        w, mus = self.tensor_factorizer.fit_transform(T)

        # Unwhiten the solutions
        # Refer to Theorem 4.3 in
        # ``Tensor Decompositions for Learning Latent Variable Models``
        # By Anandkumar, Ge, Hsu, Kakade and Telgarsky
        mu_hat = w.view(1, -1) * (B @ mus)  # [d, k]
        w_hat = 1 / (w**2)
        if self.normalize_weights:
            w_hat = F.normalize(w_hat, dim=0, p=1)
        return w_hat, mu_hat.T, torch.sqrt(sigma_sqr_est)
