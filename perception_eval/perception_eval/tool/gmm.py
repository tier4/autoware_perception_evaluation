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

from __future__ import annotations

import logging
import pickle
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from perception_eval.tool.perception_performance_analyzer import PerceptionPerformanceAnalyzer


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
        self.__models: Optional[List[GaussianMixture]] = None

    @classmethod
    def load(cls, filename: str) -> Gmm:
        """[summary]
        Load from saved model.
        Args:
            filename (str): File path.

        Returns:
            Gmm: Loaded GMM.
        """
        with open(filename, "rb") as f:
            gmm = pickle.load(f)
        return gmm

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
            x_test (Optional[numpy.ndarray]): Test data to determine number of components, in shape (N, D).
                If None, input data will be used. Defaults to None.
        """
        if self.__models is None:
            self.__models = [
                GaussianMixture(n, n_init=self.n_init, random_state=self.random_state).fit(x)
                for n in range(1, self.max_k + 1)
            ]
        else:
            self.__models = [model.fit(x) for model in self.__models]

        if x_test is None:
            x_test = x

        self.aic_list: List[float] = [m.aic(x_test) for m in self.__models]
        self.bic_list: List[float] = [m.bic(x_test) for m in self.__models]

        min_aic_idx: int = np.argmin(self.aic_list)
        min_bic_idx: int = np.argmin(self.bic_list)
        if min_aic_idx != min_bic_idx:
            logging.warning(
                f"min AIC and BIC is not same, got K={min_aic_idx + 1} and {min_bic_idx + 1}"
            )
        self.model = self.__models[min_bic_idx]

    def save(self, filename: str) -> None:
        """[summary]
        Save estimated model's parameters.

        Args:
            filename (str): Path to save model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been estimated")

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def get_gamma(self, x: np.ndarray) -> np.ndarray:
        """[summary]
        Returns gamma that describes the weight

        Args:
            x (numpy.ndarray): Input data, in shape (N, D)

        Returns:
            gamma (numpy.ndarray): Weight, in shape (K,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            raise RuntimeError(f"Invalid input shape: {x.shape}, expected (N, D)")
        x_dim: int = x.shape[-1]

        pdf = np.array(
            [
                self.pi[i]
                * multivariate_normal.pdf(
                    x,
                    self.means[i, :x_dim],
                    self.covariances[i, :x_dim, :x_dim],
                )
                for i in range(self.num_k)
            ]
        ).T
        with np.errstate(invalid="ignore", divide="ignore"):
            gamma = np.exp(np.log(pdf) - np.log(np.sum(pdf, axis=-1, keepdims=True)))
        gamma[np.isnan(gamma)] = 1.0 / self.num_k

        return gamma

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

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            raise RuntimeError(f"Invalid input shape: {x.shape}, expected (N, D)")
        x_dim: int = x.shape[-1]

        mean_x: np.ndarray = self.means[:, :x_dim]
        mean_y: np.ndarray = self.means[:, x_dim:]
        cov_xx: np.ndarray = self.covariances[:, :x_dim, :x_dim]
        cov_xy: np.ndarray = self.covariances[:, :x_dim, x_dim:]

        gamma = self.get_gamma(x)

        if kernel == "mean":
            return np.mean(
                gamma[:, :, None]
                * np.matmul(
                    (x[:, None, :] - mean_x)[:, :, None, ...],
                    np.matmul(np.linalg.inv(cov_xx), cov_xy)[None, ...],
                )[:, :, 0],
                axis=1,
            )
        elif kernel == "mode":
            i: int = np.argmax(gamma)
            return gamma[:, i][:, None] * (
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

    def plot_ic(self, filename: Optional[str] = None, show: bool = False) -> None:
        """[summary]
        Plot Information Criterion scores, which are AIC and BIC, for each number of components.
        """
        if self.aic_list is None or self.bic_list is None:
            raise RuntimeError("Model has not been estimated.")

        _, ax = plt.subplots(figsize=(6, 6))
        num_components = np.arange(1, self.max_k + 1)
        ax.plot(num_components, self.aic_list, label="AIC")
        ax.plot(num_components, self.bic_list, label="BIC")
        ax.legend(loc="best")
        ax.set_xlabel("num_k")

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()
        plt.close()


def load_sample(
    analyzer: PerceptionPerformanceAnalyzer,
    state: List[str],
    error: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load sample data
    Args:
        state (List[str]): List of state names. For example, [x, y].
        error (List[str]): List of error names. For example, [x, y].
    Returns:
        state_arr (numpy.ndarray): Array of states, in shape (N, num_state).
        error_arr (numpy.ndarray): Array of errors, in shape (N, num_error).
    """
    state_arr = np.array(analyzer.get_ground_truth(status="TP")[state])
    error_arr = np.array([analyzer.calculate_error(col) for col in error]).reshape(-1, len(error))

    # Remove nan
    not_nan = ~np.isnan(state_arr).any(1) * ~np.isnan(error_arr).any(1)
    state_arr = state_arr[not_nan]
    error_arr = error_arr[not_nan]

    return state_arr, error_arr
