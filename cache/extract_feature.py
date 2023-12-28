import os
import numpy as np
import torch
import torch.nn.functional as F


def extract_feature(model, loader, cfg, mode="train"):
    model_name = cfg.MODEL.ARCH
    dir = {'train': cfg.CACHE.TRAIN_DIR,
           'test': cfg.CACHE.TEST_DIR,
           'calib': cfg.CACHE.CALIB_DIR}
    num_classes = cfg.DATA.CLASS
    batch_size = cfg.TEST.BATCH_SIZE

    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    score, feature_list = model.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]

    if not os.path.exists(f"{dir[mode]}/model_name_feature.npy"):
        os.makedirs(dir[mode])

        features = np.zeros((len(loader.dataset), sum(featdims)))
        logits = np.zeros((len(loader.dataset), num_classes))
        labels = np.zeros(len(loader.dataset))

        model.eval()
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(loader.dataset))

            batch_logits, feature_list = model.feature_list(inputs)
            batch_features = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list],
                                       dim=1)

            features[start_ind:end_ind, :] = batch_features.data.cpu().numpy()
            labels[start_ind:end_ind] = targets.data.cpu().numpy()
            logits[start_ind:end_ind] = batch_logits.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(loader)}")

        np.save(f"{dir[mode]}/model_name_feature.npy", features)
        np.save(f"{dir[mode]}/model_name_logits.npy", logits)
        np.save(f"{dir[mode]}/model_name_labels.npy", labels)

    else:
        features = np.load(f"{dir[mode]}/model_name_feature.npy", allow_pickle=True)
        logits = np.load(f"{dir[mode]}/model_name_logits.npy", allow_pickle=True)
        labels = np.load(f"{dir[mode]}/model_name_labels.npy", allow_pickle=True)

    return features, logits, labels
