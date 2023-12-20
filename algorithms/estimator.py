import numpy as np
import torch
from tqdm import tqdm
from scipy.stats._multivariate import _PSD
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import logsumexp


def build_estimator(cfg):
    distance = cfg.TEST.DISTANCE

    if distance == "gmm":
        return GMMAtypicalityEstimator()

    elif distance == "knn":
        return KNNDistance()

    else:
        raise NotImplementedError("Unknown distance {}".format(distance))


class GMMAtypicalityEstimator:
    """
    Gaussian mixture model assuming that mixture components share the covariance matrix.
    """

    def __init__(self, pca=None, atypicality_type="class"):
        self.name = f"gmmFast_{pca}_{atypicality_type}"
        self.atypicality_type = atypicality_type
        self.class_means = None
        self.shared_cov = None
        self.shared_prec = None
        self.n_classes = None
        self.emb_dim = None
        self.cov_est = EmpiricalCovariance(assume_centered=False)
        self.shared_cov_logdet = None  # logdeterminant of the covariance for computing the pdf
        if pca is not None:
            self.pca = PCA(n_components=pca)
        else:
            self.pca = None

    def fit(self, train_feature, train_labels):
        class_means, shared_cov = [], np.zeros((train_feature.shape[1], train_feature.shape[1]))
        div_from_means = []
        if self.pca is not None:
            train_feature = self.pca.fit_transform(train_feature)

        # Compute empirical means and covariance

        for lbl in np.unique(train_labels):
            cls_feature = train_feature[train_labels == lbl]
            means = np.mean(cls_feature, axis=0, keepdims=True)
            class_means.append(means)
            div_from_mean = cls_feature - means
            div_from_means.append(div_from_mean)

        self.class_means = np.concatenate(class_means, axis=0)

        div_from_means = np.concatenate(div_from_means, axis=0)
        self.cov_est.fit(div_from_means)
        self.shared_cov = self.cov_est.covariance_

        # Use internal scipy fns for numerical stability. Too lazy to do this on my own
        psd = _PSD(self.shared_cov)
        self.shared_prec = np.dot(psd.U, psd.U.T)

        self.shared_cov_logdet = psd.log_pdet
        print("Means and convs are computed!")

        print("Inverse taken!")
        self.n_classes = len(class_means)
        self.emb_dim = self.shared_cov.shape[1]

    def _compute_mvn_logpdf(self, feature, mean, cov_log_det, prec):
        """
        Compute the pdf of the gaussian
        """
        logpart1 = - 0.5 * (self.emb_dim * np.log(2 * np.pi) + cov_log_det)
        part2 = (-1 / 2) * np.einsum("nd, nd->n", (feature - mean).dot(prec), (feature - mean))
        return logpart1 + part2

    def compute_logphat_x(self, logphatx_y, phat_y):
        loglikelihood = logsumexp(logphatx_y + np.log(phat_y), axis=1)
        return loglikelihood

    def compute_atypicality(self, feature, py_x=None):
        if self.pca is not None:
            feature = self.pca.transform(feature)

        cls_ps = []
        for idx in tqdm(range(self.n_classes)):
            cls_mean = self.class_means[idx]
            cls_p = self._compute_mvn_logpdf(feature, mean=cls_mean, cov_log_det=self.shared_cov_logdet,
                                             prec=self.shared_prec)
            cls_ps.append(cls_p)

        cls_ps = np.vstack(cls_ps).T
        logphat_x = self.compute_logphat_x(cls_ps, np.ones(self.n_classes) / self.n_classes)
        if self.atypicality_type == "class":
            # Returns - \max_c \hat{p}(X|Y=c)
            return -np.max(cls_ps, axis=1)
        elif self.atypicality_type == "all_class":
            # Returns - \max_c \hat{p}(X|Y=c)
            return -cls_ps
        else:
            # Returns - \hat{p}(X)
            return -logphat_x


class KNNDistance:
    """
    KNN Distance for an estimate of density.
    """

    def __init__(self, k=10, case="worst", return_all=False):
        """
        k: # of nearest neighbors
        case: Whether to use the average distance or the worst case distance.
        """
        self.name = f"kNN_{k}_{case}"
        self.n_neighbors = k
        self.case = case
        self.n_classes = None
        self.knns = {}
        self.return_all = return_all

    def fit(self, train_feature, train_labels):
        for lbl in np.unique(train_labels):
            cls_feature = train_feature[train_labels == lbl]
            self.knns[lbl] = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.knns[lbl].fit(cls_feature, train_labels[train_labels == lbl])

        self.n_classes = len(self.knns)

    def compute_atypicality(self, feature, py_x=None):
        all_scores = []
        for idx in tqdm(range(self.n_classes)):
            cls_neigh_dist, _ = self.knns[idx].kneighbors(feature, return_distance=True)
            if self.case == "worst":
                all_scores.append(cls_neigh_dist[:, -1:])
            elif self.case == "average":
                all_scores.append(np.mean(cls_neigh_dist, axis=1, keepdims=True))
            else:
                raise ValueError("Invalid case")

        all_scores = np.concatenate(all_scores, axis=1)
        if self.return_all:
            return all_scores
        else:
            return np.min(all_scores, axis=1)


class KNN_estimator:

    def __init__(self, cfg):
        self.classes = cfg.DATA.CLASS
        self.k = cfg.TEST.K

    def fit(self, features, labels):
        self.features = torch.tensor(features).cuda()
        self.labels = torch.tensor(labels).cuda()

    def compute_atypicality(self, features):
        features = torch.tensor(features).cuda()
        atypicality_list = []

        for i in tqdm(range(len(features))):
            distance = torch.norm(features[i] - self.features, dim=1)
            top_k_distances = torch.sort(distance)[:self.k]
            atypicality = np.exp(np.sum(top_k_distances) / self.k)
            atypicality_list.append(atypicality)

        return atypicality_list
