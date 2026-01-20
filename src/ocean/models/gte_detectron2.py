from __future__ import annotations

"""Detectron2 GTE-inspired joint table+cell detector.

This module is a cleaned-up extraction of the logic in `final_version1_0.ipynb`:
- One model predicts both *tables* and *cells* (two classes).
- Adds a *cell containment* auxiliary loss: predicted cells should lie within a predicted table.

NOTE
----
Detectron2 is not listed as a regular dependency because installation is platform/CUDA-dependent.
You are expected to install it separately (then install this repo with the `ml` extras).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

TABLE_CAT_ID = 0
CELL_CAT_ID = 1


@dataclass
class ContainmentConfig:
    weight: float = 1.0
    iou_thresh: float = 0.5


def build_model_and_trainer(cfg_path: str | None = None):
    """Utility entrypoint for experimentation.

    Returns (cfg, trainer)

    This is intentionally a thin wrapper: your real training script should live
    in `ocean/cli/train_detectron2.py` so you can version your experiments.
    """
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    if cfg_path:
        cfg.merge_from_file(cfg_path)
    trainer = DefaultTrainer(cfg)
    return cfg, trainer


def _containment_loss(
    pred_boxes, pred_classes, pred_scores, *, iou_thresh: float
) -> "torch.Tensor":
    """Penalize predicted cell boxes that are not inside any predicted table.

    We compute IoU between each predicted *cell* box and each predicted *table* box.
    A cell is considered 'contained' if max IoU >= iou_thresh.

    Loss = mean(ReLU(iou_thresh - max_iou)) over predicted cells.

    This is a simple, stable surrogate that matches the *intent* of the notebook.
    """
    import torch
    from detectron2.structures import pairwise_iou, Boxes

    device = pred_boxes.tensor.device

    table_mask = pred_classes == TABLE_CAT_ID
    cell_mask = pred_classes == CELL_CAT_ID

    if table_mask.sum() == 0 or cell_mask.sum() == 0:
        return torch.zeros((), device=device)

    table_boxes = Boxes(pred_boxes.tensor[table_mask])
    cell_boxes = Boxes(pred_boxes.tensor[cell_mask])

    ious = pairwise_iou(cell_boxes, table_boxes)  # (num_cells, num_tables)
    max_iou, _ = ious.max(dim=1)

    # Optional: weight by confidence to focus loss on higher-score cells.
    cell_scores = pred_scores[cell_mask].clamp_min(1e-6)

    penalty = (iou_thresh - max_iou).clamp_min(0.0)
    loss = (penalty * cell_scores).mean()
    return loss


def add_containment_loss_to_roi_heads(model, containment: ContainmentConfig):
    """Monkey-patch a Detectron2 GeneralizedRCNN model to add containment loss.

    Why patch?
    - Keeps your config/architecture mostly standard.
    - Avoids needing to fully re-register a custom META_ARCH for simple experiments.

    If you want a fully custom meta-architecture (as in the notebook), build it here.
    """
    import torch

    old_forward = model.forward

    def forward_with_containment(batched_inputs):  # noqa: ANN001
        out = old_forward(batched_inputs)

        # During training, Detectron2 returns a dict of losses.
        if isinstance(out, dict) and model.training:
            # Try to access proposals / predictions is tricky across versions.
            # So we compute containment from *final* Instances in the predictor style
            # by running the model's inference branch on the same inputs.
            # This adds compute but is robust and easy to reason about.
            with torch.no_grad():
                model.eval()
                preds = old_forward(batched_inputs)
                model.train()

            # preds in eval mode is typically a list of dicts with "instances".
            loss_accum = None
            n = 0
            if isinstance(preds, list):
                for p in preds:
                    inst = p.get("instances")
                    if inst is None:
                        continue
                    n += 1
                    loss_i = _containment_loss(
                        inst.pred_boxes,
                        inst.pred_classes,
                        inst.scores,
                        iou_thresh=containment.iou_thresh,
                    )
                    loss_accum = loss_i if loss_accum is None else (loss_accum + loss_i)

            if n and loss_accum is not None:
                out["loss_containment"] = containment.weight * (loss_accum / n)
            else:
                out["loss_containment"] = torch.zeros((), device=model.device)

        return out

    model.forward = forward_with_containment
    logger.info(
        "Containment loss enabled (weight=%s, iou_thresh=%s)",
        containment.weight,
        containment.iou_thresh,
    )
    return model
