import random
import os
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax


def set_seed(cfg):
    seed = cfg.TEST.SEED
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_fig_records(info, N_groups=5, **metadata):
    records = []
    probs = info["probs"]
    labels = info["labels"]
    atypicality = info["atypicality"].flatten()

    quantiles = np.linspace(0, 1, N_groups)
    for conf in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
        conf_lower = conf
        conf_upper = conf + 0.1
        for q_lower, q_higher in zip(quantiles[:-1], quantiles[1:]):
            vs = np.quantile(atypicality, q=[q_lower, q_higher])
            # Control for the start
            if q_lower == 0:
                vs[0] = -np.inf

            mask = ((atypicality <= vs[1]) & (atypicality > vs[0]) & (np.max(probs, 1) >= conf_lower)
                    & (np.max(probs, 1) <= conf_upper))
            group_probs = probs[mask]
            group_lbls = labels[mask]
            group_atypicality = atypicality[mask]

            record = {
                "Accuracy": (np.argmax(group_probs, axis=1) == group_lbls).mean(),
                "MeanAtypicality": group_atypicality.mean(),
                "confidence": f"{conf_lower}-{conf_upper}"
            }
            records.append(record)
    return records


def evaluate(labels, logits, atypicality, N_groups=10):
    prob_info = {
        "probs": softmax(logits, 1),
        "atypicality": atypicality,
        "labels": labels
    }
    all_records = get_fig_records(prob_info, N_groups=N_groups)
    data = pd.DataFrame(all_records)
    return data
