import torch

from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator


class SegmentationEvaluator(BaseEvaluator):
    """
    Pixel-level binary segmentation evaluator.

    Accumulates TP, FP, FN, TN counts across all samples and computes
    metrics that are robust to the severe class imbalance common in
    medical segmentation, where labelled (foreground) pixels are a tiny
    fraction of the total pixel population.

    Why not accuracy?
    -----------------
    In a typical coronary calcium CT slice, foreground (calcium) pixels
    may represent less than 1% of all pixels.  A model that predicts
    all-negative achieves >99% pixel accuracy while detecting nothing.
    Dice (F1) and IoU (Jaccard) are computed entirely from the foreground
    class and are therefore unaffected by the size of the true-negative
    background.  Accuracy is still reported for completeness but is
    labelled explicitly as unreliable in sparse-label settings.

    Metrics produced by ``aggregate()``
    ------------------------------------
    tp, fp, fn, tn         Raw pixel counts.
    precision              TP / (TP + FP)
    recall                 TP / (TP + FN)   [also called sensitivity]
    dice                   2·TP / (2·TP + FP + FN)  [identical to F1]
    iou                    TP / (TP + FP + FN)       [Jaccard index]
    accuracy               (TP + TN) / total pixels  [unreliable when
                           foreground pixels are sparse — reported for
                           completeness only]

    Parameters
    ----------
    threshold : float
        Decision boundary applied to raw model logits.  Defaults to
        0.0, which is equivalent to sigmoid > 0.5.  Adjust for
        operating points that favour precision over recall or vice versa.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    # --------------------------------------------------
    # BaseEvaluator interface
    # --------------------------------------------------

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate pixel counts for one sample.

        Parameters
        ----------
        pred : Tensor
            Raw model logits of any shape.  Binarised at ``self.threshold``.
        target : Tensor
            Ground-truth mask of the same shape.  Treated as positive
            where value > 0.5.
        """
        pred_bin = (pred > self.threshold).long().view(-1)
        y_bin    = (target > 0.5).long().view(-1)

        self._tp += int(((pred_bin == 1) & (y_bin == 1)).sum())
        self._fp += int(((pred_bin == 1) & (y_bin == 0)).sum())
        self._fn += int(((pred_bin == 0) & (y_bin == 1)).sum())
        self._tn += int(((pred_bin == 0) & (y_bin == 0)).sum())

    def aggregate(self) -> dict:
        """Return all segmentation metrics as a flat dict."""

        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        total = tp + fp + fn + tn

        precision = tp / (tp + fp)           if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn)           if (tp + fn) > 0 else float("nan")
        dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        iou       = tp / (tp + fp + fn)      if (tp + fp + fn) > 0 else float("nan")
        accuracy  = (tp + tn) / total        if total > 0 else float("nan")

        return {
            # raw counts
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # foreground-focused metrics (class-imbalance robust)
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            # included for completeness; unreliable when foreground is sparse
            "accuracy": accuracy,
        }
