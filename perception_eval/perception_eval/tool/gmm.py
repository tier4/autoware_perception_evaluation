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

import logging
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class Gmm:
    """This is a wrapper class of sklearn's GaussianMixture.

    Attributes:
        self.max_k (int)
        self.n_init (int)
        self.random_state (int)
        self.model (Optional[GaussianMixture])

    Properties:
        num_k (int): Number of clusters K.
        pi (numpy.ndarray): Weight of the GMM, in shape of (K,).
        means (numpy.ndarray): Mean value of the GMM, in shape (K, D).
        covariances (numpy.ndarray): Variance-covariance matrix of the GMM, in shape of (K, D, D).
    """

    eps: float = np.spacing(1)

    def __init__(self, max_k: int, n_init: int = 1, random_state: int = 1234) -> None:
        """
        Args:
            max_k (int): Maximum number of clusters K.
            n_init (int)
            random_state (int)
        """
        self.max_k: int = max_k
        self.n_init: int = n_init
        self.random_state: int = random_state
        self.model: Optional[GaussianMixture] = None

    @property
    def num_k(self) -> int:
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")
        return self.model.n_components

    @property
    def pi(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")
        return self.model.weights_

    @property
    def means(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")
        return self.model.means_

    @property
    def covariances(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")
        return self.model.covariances_

    def fit(
        self,
        x: np.ndarray,
        x_test: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]
        Fitting parameters for GMM.

        Args:
            x (numpy.ndarray): Input data, in shape of (N, D).
            max_iter (int): Maximum number of updates.
            threshold (float): Threshold of convergence condition. Defaults to 1e-3.
        """
        models = [
            GaussianMixture(n, n_init=self.n_init, random_state=self.random_state).fit(x)
            for n in range(1, self.max_k + 1)
        ]

        if x_test is None:
            x_test = x

        self.aic_list: List[float] = [m.aic(x_test) for m in models]
        self.bic_list: List[float] = [m.bic(x_test) for m in models]

        min_aic_idx: int = np.argmin(self.aic_list)
        min_bic_idx: int = np.argmin(self.bic_list)
        if min_aic_idx != min_bic_idx:
            logging.warning(
                "min AIC and BIC is not same,",
                f"got K={min_aic_idx} and {min_bic_idx}",
            )
        self.model = models[min_bic_idx]

    def predict(
        self,
        x: np.ndarray,
        kernel: str = "mean",
    ) -> np.ndarray:
        """[summary]
        Predict means of posteriors.

        Args:
            x (numpy.ndarray)
            kernel (str): mean or mode. Defaults to mean.

        Returns:
            numpy.ndarray: Predicted means.
        """
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")

        if x.ndim != 1:
            # TODO: add support of data in shape (N, D)
            raise RuntimeError("Expected data in shape (D,)")
        x_dim: int = x.shape[-1]

        mean_x: np.ndarray = self.means[:, :x_dim]
        mean_y: np.ndarray = self.means[:, x_dim:]
        cov_xx: np.ndarray = self.covariances[:, :x_dim, :x_dim]
        cov_xy: np.ndarray = self.covariances[:, :x_dim, x_dim:]

        pdf = np.array(
            [
                self.pi[i] * multivariate_normal.pdf(x, mean_x[i], cov_xx[i])
                for i in range(self.num_k)
            ]
        )
        gamma = np.exp(
            np.log(pdf) - (np.log(np.sum(pdf, axis=x.ndim - 1, keepdims=True) + self.eps))
        )
        gamma[np.isnan(gamma)] = 1.0 / self.num_k

        if kernel == "mean":
            values: List[np.ndarray] = [
                gamma[i]
                * (
                    mean_y[i]
                    + np.matmul(
                        (x - mean_x[i]),
                        np.matmul(
                            np.linalg.inv(cov_xx[i]),
                            cov_xy[i],
                        ),
                    )
                )
                for i in range(self.num_k)
            ]
            return np.mean(values, axis=0)
        elif kernel == "mode":
            i = np.argmax(gamma)
            return gamma[i] * (
                mean_y[i]
                + np.matmul(
                    (x - mean_x[i]),
                    np.matmul(
                        np.linalg.inv(cov_xx[i]),
                        cov_xy[i],
                    ),
                )
            )
        else:
            raise ValueError(f"Expected kernel mean or mode, but got {kernel}")

    def predict_label(self, x: np.ndarray) -> np.ndarray:
        """[summary]
        Predict cluster of input.

        Args:
            x (numpy.ndarray)

        Returns:
            numpy.ndarray: Array of labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been estimated.")
        return self.model.predict(x)

    def plot(self) -> None:
        """[summary]
        Plot AIC and BIC.
        """
        if self.aic_list is None or self.bic_list is None:
            raise RuntimeError("Model has not been estimated.")

        _, ax = plt.subplots(figsize=(6, 6))
        num_components = np.arange(1, self.max_k + 1)
        ax.plot(num_components, self.aic_list, label="AIC")
        ax.plot(num_components, self.bic_list, label="BIC")
        ax.legend(loc="best")
        ax.set_xlabel("num_k")

        plt.show()
        plt.close()
