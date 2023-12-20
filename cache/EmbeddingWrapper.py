import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_test(lbls, acts, logits, split=0.2, seed=1):
    indices = np.arange(len(lbls))
    lbls_1, lbls_2, acts_1, acts_2, logits_1, logits_2, train_idx, test_idx = train_test_split(lbls, acts, logits,
                                                                                               indices,
                                                                                               test_size=1 - split,
                                                                                               random_state=seed,
                                                                                               stratify=lbls)
    return lbls_1, acts_1, logits_1, train_idx, lbls_2, acts_2, logits_2, test_idx


class EmbeddingWrapper:
    def __init__(self, backbone, model, model_name):
        self.backbone = backbone
        self.model = model
        self.model_name = model_name

    @torch.no_grad()
    def get_outputs(self, loader):
        """Runs and returns models embeddings, labels, and logits for the given datasets."""
        features = []
        labels = []
        logits = []
        for image, label in tqdm(loader):
            # get feature
            batch_feature = self.backbone(image).view(image.shape[0], -1)
            features.append(batch_feature.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            logits.append(self.model(image).detach().cpu().numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        return features, labels, logits

    @torch.no_grad()
    def run_and_cache_outputs(self, dataloader, output_dir="cache/caches", mode="train"):
        """
        If the experiment files (embeddings, labels, logits) already exist, load them. Otherwise, run the models and cache the outputs.
        """
        if mode == "train":
            output_dir = output_dir + "/train"
            features_file = os.path.join(output_dir, f"{self.model_name}_features.npy")
            labels_file = os.path.join(output_dir, f"{self.model_name}_labels.npy")
            logits_file = os.path.join(output_dir, f"{self.model_name}_logits.npy")
        elif mode == "test":
            output_dir = output_dir + "/test"
            features_file = os.path.join(output_dir, f"{self.model_name}_features.npy")
            labels_file = os.path.join(output_dir, f"{self.model_name}_labels.npy")
            logits_file = os.path.join(output_dir, f"{self.model_name}_logits.npy")
        elif mode == "calib":
            output_dir = output_dir + "/calib"
            features_file = os.path.join(output_dir, f"{self.model_name}_features.npy")
            labels_file = os.path.join(output_dir, f"{self.model_name}_labels.npy")
            logits_file = os.path.join(output_dir, f"{self.model_name}_logits.npy")

        if os.path.exists(logits_file):
            print(f"Found: {logits_file}, loading.")
            features = np.load(features_file)
            labels = np.load(labels_file)
            logits = np.load(logits_file)
        else:
            print(f"Not found: {logits_file}, extracting.")
            features, labels, logits = self.get_outputs(dataloader)

            if not os.path.exists("cache/caches/train"):
                os.makedirs("cache/caches/train")
            if not os.path.exists("cache/caches/test"):
                os.makedirs("cache/caches/test")
                if not os.path.exists("cache/caches/calib"):
                    os.makedirs("cache/caches/calib")

            np.save(features_file, features)
            np.save(labels_file, labels)
            np.save(logits_file, logits)
        return features, logits, labels
