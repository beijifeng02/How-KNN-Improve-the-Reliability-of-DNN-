import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


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
