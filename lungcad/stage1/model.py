"""Stage 1 — RetinaNet 3D (MONAI monai.apps.detection), following the official LUNA16 detection tutorial.

Backbone: 3D ResNet-50 (bottleneck) + FPN, returned layers (1,2) -> 3 feature levels with strides 4/8/16 in-plane.
Axis convention: our volumes are (z, y, x) with spacing (1.25, 0.703, 0.703) mm, so everything that is
(x, y, z) in the tutorial is permuted to (z, y, x) here: conv1 stride (1,2,2), anchors (z,y,x).
Anchor base shapes (voxels) cover the LUNA16 nodule diameter range 3–33 mm: [4,6,6],[5,8,8],[6,10,10],[8,16,16],[12,24,24]
at the finest level, scaled x2 / x4 at the coarser levels.
Losses: MONAI RetinaNetDetector defaults, as trained in S3-S5: binary cross-entropy on the anchors chosen by the
hard-negative sampler (classification) + Smooth-L1, beta 1/9, on encoded box offsets (regression). Matcher: ATSS.
(Earlier text here said focal + GIoU; that was never what this code trained.)
"""
from __future__ import annotations
import torch
from monai.networks.nets import resnet
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

DEFAULT = {
    "returned_layers": [1, 2],
    "conv1_t_stride": [1, 2, 2],                       # (z, y, x)
    "base_anchor_shapes": [[4, 6, 6], [5, 8, 8], [6, 10, 10], [8, 16, 16], [12, 24, 24]],   # (z, y, x) voxels
    "backbone": "resnet50",
    "atss_num_candidates": 4,
    "hns_batch_size_per_image": 64, "hns_positive_fraction": 0.3, "hns_pool_size": 20, "hns_min_neg": 16,
    "score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 300, "topk_candidates_per_level": 1000,
}


def build_detector(cfg: dict | None = None, device="cuda") -> RetinaNetDetector:
    c = {**DEFAULT, **(cfg or {})}
    returned_layers = list(c["returned_layers"])
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2 ** l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=c["base_anchor_shapes"])
    conv1_t_stride = tuple(c["conv1_t_stride"])
    conv1_t_size = tuple(max(7, 2 * s + 1) for s in conv1_t_stride)
    if c["backbone"] == "resnet50":
        block, layers = resnet.ResNetBottleneck, [3, 4, 6, 3]
    elif c["backbone"] == "resnet18":
        block, layers = resnet.ResNetBlock, [2, 2, 2, 2]
    elif c["backbone"] == "resnet34":
        block, layers = resnet.ResNetBlock, [3, 4, 6, 3]
    else:
        raise ValueError(c["backbone"])
    backbone = resnet.ResNet(block=block, layers=layers, block_inplanes=resnet.get_inplanes(), n_input_channels=1,
                             conv1_t_stride=conv1_t_stride, conv1_t_size=conv1_t_size, spatial_dims=3)
    feature_extractor = resnet_fpn_feature_extractor(backbone=backbone, spatial_dims=3, pretrained_backbone=False,
                                                     trainable_backbone_layers=None, returned_layers=returned_layers)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    net = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=num_anchors, feature_extractor=feature_extractor, size_divisible=size_divisible)
    det = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)
    det.set_atss_matcher(num_candidates=c["atss_num_candidates"], center_in_gt=False)
    det.set_hard_negative_sampler(batch_size_per_image=c["hns_batch_size_per_image"], positive_fraction=c["hns_positive_fraction"],
                                  pool_size=c["hns_pool_size"], min_neg=c["hns_min_neg"])
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(score_thresh=c["score_thresh"], topk_candidates_per_level=c["topk_candidates_per_level"],
                                    nms_thresh=c["nms_thresh"], detections_per_img=c["detections_per_img"])
    det.size_divisible_zyx = size_divisible
    return det


def set_inferer(det: RetinaNetDetector, roi_size, overlap=0.25, sw_batch_size=2, mode="gaussian"):
    det.set_sliding_window_inferer(roi_size=list(roi_size), overlap=overlap, sw_batch_size=sw_batch_size, mode=mode,
                                   cval=0.0, device="cpu", sw_device="cuda", progress=False)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
