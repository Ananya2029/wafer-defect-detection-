"""Inference helpers and Grad-CAM for the wafer CNN."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from wafer.data import LABELS, ROOT, to_channels
from wafer.model import WaferCNN

MODEL_DIR = ROOT / "models"


def load_model():
    model = WaferCNN()
    model.load_state_dict(torch.load(MODEL_DIR / "cnn.pt", map_location="cpu"))
    model.eval()
    thresholds = np.array(json.loads((MODEL_DIR / "thresholds.json").read_text())["thresholds"])
    return model, thresholds


def predict_one(model, thresholds, wafer_map):
    """Probabilities and detected patterns for a single 52x52 map (0 blank, 1 good, 2 defect)."""
    x = torch.from_numpy(to_channels(wafer_map[None]))
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0].numpy()
    detected = [LABELS[j] for j in range(len(LABELS)) if prob[j] >= thresholds[j]]
    return prob, detected


def grad_cam(model, wafer_map, label_index, layer=4):
    """Grad-CAM heat map (52x52, 0..1) showing which dies drove the score for one pattern.

    layer=4 is the 13x13 conv block: fine enough to localise defects on a 52x52 map
    (the final 6x6 block gives maps too coarse to be useful)."""
    x = torch.from_numpy(to_channels(wafer_map[None])).requires_grad_(False)
    store = {}
    target_layer = model.features[layer]
    h1 = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.zero_grad()
        logits = model(x)
        logits[0, label_index].backward()
    finally:
        h1.remove()
        h2.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)          # importance of each channel
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=wafer_map.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    cam = cam * (wafer_map > 0)                                     # only inside the wafer
    return cam / cam.max() if cam.max() > 0 else cam
