# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional

import numpy as np
from scipy.linalg import pinvh
from scipy.stats import multivariate_normal


class Gmm:
    """EM algorithm for Gaussian mixture model.

    Attributes:
        num_k (int): Number of clusters K.
        num_n (int): Number of input data N.
        num_d (int): Number of dimensions D.
        mu (numpy.ndarray): Mean value of the GMM, in shape (K, D).
        sigma (numpy.ndarray): Variance-covariance matrix of the GMM, in shape of (K, D, D).
        pi (numpy.ndarray): Weight of the GMM, in shape of (K,).
        gamma (numpy.ndarray): Gamma value, in shape of (D, K).
    """

    # eps: float = np.spacing(1)
    eps: float = 1e-6

    def __init__(self, max_k: int) -> None:
        """
        Args:
            max_k (int): Maximum number of clusters K.
        """
        self.__max_k: int = max_k

        self.__num_k: int = 1
        self.__data: Optional[np.ndarray] = None
        self.__num_n: Optional[int] = None
        self.__num_d: Optional[int] = None
        self.__mu: Optional[np.ndarray] = None
        self.__sigma: Optional[np.ndarray] = None
        self.__pi: Optional[np.ndarray] = None
        self.__gamma: Optional[np.ndarray] = None

        self.__bic_list: List[float] = []
        self.__mu_list: List[np.ndarray] = []
        self.__sigma_list: List[np.ndarray] = []
        self.__pi_list: List[np.ndarray] = []
        self.__gamma_list: List[np.ndarray] = []

    @property
    def num_k(self) -> int:
        """
        Returns:
            int: Number of clusters K.
        """
        return self.__num_k

    @property
    def num_n(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: Number of input data N.
        """
        return self.__num_n

    @property
    def num_d(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: Number of dimensions D.
        """
        return self.__num_d

    @property
    def mu(self) -> Optional[np.ndarray]:
        """
        Returns:
            Optional[numpy.ndarray]: Mean value of the GMM, in shape of (K, D).
        """
        return self.__mu

    @property
    def sigma(self) -> Optional[np.ndarray]:
        """
        Returns:
            Optional[numpy.ndarray]: Variance-covariance matrix of the GMM, in shape of (K, D, D).
        """
        return self.__sigma

    @property
    def pi(self) -> Optional[np.ndarray]:
        """
        Returns:
            Optional[numpy.ndarray]: Weight of the GMM, in shape of (K,).
        """
        return self.__pi

    @property
    def gamma(self) -> Optional[np.ndarray]:
        """
        Returns:
            Optional[numpy.ndarray]: Gamma value, in shape of (D, K).
        """
        return self.__gamma

    @property
    def data(self) -> Optional[np.ndarray]:
        """
        Returns:
            Optional[numpy.ndarray]: Input data, in shape of (N, D).
        """
        return self.__data

    @property
    def is_initialized(self) -> bool:
        """
        Returns:
            bool: Whether parameters have been initialized.
        """
        return (
            self.__num_n
            and self.__num_d
            and self.__mu
            and self.__sigma
            and self.__pi
            and self.__gamma
        )

    def __init_params(self, x: np.ndarray) -> None:
        """[summary]
        Initialize the parameters.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).
        """
        self.__data = x
        self.__num_n, self.__num_d = x.shape
        self.__mu = np.random.randn(self.num_k, self.num_d)
        self.__sigma = np.tile(np.eye(self.num_d), reps=(self.num_k, 1, 1))
        self.__pi = np.ones(self.num_k) / self.num_k
        self.__gamma = np.random.randn(self.num_d, self.num_k)

    def __calc_pdf(self, x: np.ndarray) -> np.ndarray:
        """[summary]
        Returns the log-likelihood of the D-dimensional Gaussian mixed distribution at N data.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).

        Returns:
            numpy.ndarray: Probability density function, in shape of (N, K).
        """
        return np.array(
            [
                self.pi[k] * multivariate_normal.pdf(x, mean=self.mu[k], cov=self.sigma[k])
                for k in range(self.num_k)
            ]
        ).T

    def __calc_bic(self, x: np.ndarray) -> float:
        """[summary]
        Calculate Bayesian Information Criterion."""
        log_lh: float = np.mean(np.log(self.__calc_pdf(x).sum(axis=1) + self.eps))
        log_n: float = np.log(self.__num_n)
        return -2 * log_lh + log_n

    def __e_step(self, x: np.ndarray) -> None:
        """[summary]
        Execute the E-step of EM algorithm.
        This method optimizes self.gamma.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).
        """
        # Calculate the responsibilities in log domain
        pdf: np.ndarray = self.__calc_pdf(x)
        log_r: np.ndarray = np.log(pdf) - np.log(np.sum(pdf, 1, keepdims=True) + self.eps)
        # Modify to exponential domain
        gamma: np.ndarray = np.exp(log_r)
        # Replace the nan elements
        gamma[np.isnan(gamma)] = 1.0 / self.num_k
        # Update the optimized responsibility
        self.__gamma = gamma

    def __m_step(self, x: np.ndarray) -> None:
        """[summary]
        Execute M-step of EM algorithm.
        This method optimizes self.pi, self.mu and self.sigma.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).
        """
        num_k: np.ndarray = np.sum(self.gamma, axis=0)
        self.__pi = num_k / self.num_n
        self.__mu = (self.gamma.T @ x) / (num_k[:, None] + self.eps)  # (K, D)

        gammas = np.tile(
            self.gamma[:, :, None],
            reps=(1, 1, self.num_d),
        ).transpose(1, 2, 0)

        xs = np.tile(
            x[:, :, None],
            reps=(1, 1, self.num_k),
        ).transpose(2, 1, 0)
        mus = np.tile(
            self.mu[:, :, None],
            reps=(1, 1, self.num_n),
        )
        err: np.ndarray = xs - mus
        self.__sigma = ((gammas * err) @ err.transpose(0, 2, 1)) / (num_k[:, None, None] + self.eps)

    def fit(
        self,
        x: np.ndarray,
        max_iter: int,
        threshold: float = 1e-3,
    ) -> None:
        """[summary]
        Fitting parameters for GMM.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).
            max_iter (int): Maximum number of updates.
            threshold (float): Threshold of convergence condition. Defaults to 1e-3.
        """
        self.__data = x
        for num_k in range(1, self.__max_k + 1):
            self.__num_k = num_k
            self.__init_params(x)
            # Calculate log likelihood
            log_lh: float = np.mean(np.log(self.__calc_pdf(x).sum(axis=1) + self.eps))
            for _ in range(max_iter):
                self.__e_step(x)
                self.__m_step(x)
                # Calculate updated log likelihood
                new_log_lh: float = np.mean(np.log(self.__calc_pdf(x).sum(axis=1) + self.eps))
                if np.abs(log_lh - new_log_lh) < threshold:
                    break
                log_lh = new_log_lh

            self.__bic_list.append(self.__calc_bic(x))
            self.__mu_list.append(self.mu)
            self.__sigma_list.append(self.sigma)
            self.__pi_list.append(self.pi)
            self.__gamma_list.append(self.gamma)

        min_bic_idx: int = np.argmin(self.__bic_list)
        self.__mu = self.__mu_list[min_bic_idx]
        self.__sigma = self.__sigma_list[min_bic_idx]
        self.__pi = self.__pi_list[min_bic_idx]
        self.__gamma = self.__gamma_list[min_bic_idx]
        self.__num_k = min_bic_idx + 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """[summary]
        Predict means of posteriors.

        Args:
            x (numpy.ndarray)
        """
        # # TODO
        # if not self.is_initialized:
        #     raise RuntimeError("Parameters have not initialized yet")

        # indices: np.ndarray = np.arange(x.shape[0], dtype=int)
        # num_sample = len(x)

        # inv_indices = self.__invert_indices(indices)
        # reg_coefficients = np.empty((self.num_k, len(inv_indices), len(indices)))

        # marginal_norm_facts = np.empty(self.num_k)
        # marginal_exponents = np.empty((num_sample, self.num_k))

        # for k in range(self.num_k):
        #     reg_coefficients[k] = self.__regression_coefficients(
        #         self.sigma[k],
        #         inv_indices,
        #         indices,
        #     )
        #     pdf = self.__calc_pdf(x)  # (N, K)
        pass

    def visualize(self) -> None:
        """[summary]
        Visualize estimated GMM as a heatmap.
        """
        # # TODO
        # if not self.is_initialized:
        #     raise RuntimeError("Parameters have not initialized yet")

        # if self.num_d > 3:
        #     raise ValueError(f"Cannot visualize {self.num_d}-dimensions data.")
        pass

    def __invert_indices(self, indices):
        inv = np.ones(self.num_d, dtype=bool)
        inv[indices] = False
        (inv,) = np.where(inv)
        return inv

    @staticmethod
    def __regression_coefficients(covariance, i_out, i_in, cov_12=None) -> np.ndarray:
        """Compute regression coefficients to predict conditional distribution.
        Args:
            covariance (numpy.ndarray): Covariance of MVN, in shape of (D, D).
            i_out (numpy.ndarray): Output feature indices, in shape of (i_out,)
            i_in (numpy.ndarray): Input feature indices, in shape of (i_in,)
            cov_12 (numpy.ndarray): Precomputed block of the covariance matrix between input features and output features.
                in shape of (i_out, i_in). Defaults to None.
        Returns:
            numpy.array: shape (i_out, i_in).
                Regression coefficients. These can be used to compute the mean of the
                conditional distribution as
                mean[i1] + regression_coefficients.dot((X - mean[i2]).T).T
        """
        if cov_12 is None:
            cov_12 = covariance[np.ix_(i_out, i_in)]
        cov_22 = covariance[np.ix_(i_in, i_in)]
        inv_22 = pinvh(cov_22)
        return cov_12.dot(inv_22)


def _safe_probability_density(norm_factors, exponents):
    """Compute numerically safe probability densities of a GMM.
    The probability density of individual Gaussians in a GMM can be computed
    from a formula of the form
    q_k(X=x) = p_k(X=x) / sum_l p_l(X=x)
    where p_k(X=x) = c_k * exp(exponent_k) so that
    q_k(X=x) = c_k * exp(exponent_k) / sum_l c_l * exp(exponent_l)
    Instead of using computing this directly, we implement it in a numerically
    more stable version that works better for very small or large exponents
    that would otherwise lead to NaN or division by 0.
    The following expression is mathematically equal for any constant m:
    q_k(X=x) = c_k * exp(exponent_k - m) / sum_l c_l * exp(exponent_l - m),
    where we set m = max_l exponents_l.
    Parameters
    ----------
    norm_factors : array, shape (n_components,)
        Normalization factors of individual Gaussians
    exponents : array, shape (n_samples, n_components)
        Exponents of each combination of Gaussian and sample
    Returns
    -------
    p : array, shape (n_samples, n_components)
        Probability density of each sample
    """
    m = np.max(exponents, axis=1)[:, np.newaxis]
    p = norm_factors[np.newaxis] * np.exp(exponents - m)
    p /= np.sum(p, axis=1)[:, np.newaxis]
    return p
