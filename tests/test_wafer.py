"""Tests for data handling, features, the CNN and the explainer.

    python -m pytest -q
"""
import numpy as np
import pytest
import torch

from wafer.data import LABELS, combo_id, deduplicate, load_splits, to_channels
from wafer.features import feature_matrix
from wafer.model import WaferCNN, random_dihedral


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_no_wafer_in_both_train_and_test(splits):
    train = {x.tobytes() for x in splits["train"][0]}
    assert not any(x.tobytes() in train for x in splits["test"][0])


def test_split_sizes_and_all_combinations_present(splits):
    sizes = [len(splits[k][0]) for k in ("train", "val", "test")]
    assert sum(sizes) == 38015 - splits["n_duplicates_removed"]
    assert len(np.unique(combo_id(splits["test"][1]))) == 38


def test_deduplicate_keeps_first_copy():
    X = np.array([[[1]], [[2]], [[1]]], dtype=np.uint8)
    y = np.array([[0], [1], [0]], dtype=np.float32)
    Xd, yd, n = deduplicate(X, y)
    assert n == 1 and len(Xd) == 2 and Xd[0, 0, 0] == 1


def test_channels_and_features_shapes(splits):
    X = splits["test"][0][:3]
    ch = to_channels(X)
    assert ch.shape == (3, 2, 52, 52) and set(np.unique(ch)) <= {0.0, 1.0}
    assert feature_matrix(X).shape[0] == 3


def test_cnn_output_and_augmentation():
    x = torch.rand(4, 2, 52, 52)
    assert WaferCNN()(x).shape == (4, len(LABELS))
    assert random_dihedral(x).shape == x.shape


def test_trained_model_predicts_and_explains(splits):
    from wafer.explain import grad_cam, load_model, predict_one
    model, thr = load_model()
    X, y = splits["test"]
    i = int(np.nonzero(y[:, LABELS.index("Center")] == 1)[0][0])
    prob, detected = predict_one(model, thr, X[i])
    assert prob.shape == (8,) and "Center" in detected
    cam = grad_cam(model, X[i], LABELS.index("Center"))
    assert cam.shape == (52, 52) and 0 <= cam.min() and cam.max() <= 1
    normal = int(np.nonzero(y.sum(1) == 0)[0][0])
    assert predict_one(model, thr, X[normal])[1] == []
