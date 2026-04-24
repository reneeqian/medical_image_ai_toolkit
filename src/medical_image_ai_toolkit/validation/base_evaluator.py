from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    Abstract interface for per-sample evaluation during validation.

    The ModelTestingPipeline calls ``update()`` once per sample as it
    streams through the test partition, then calls ``aggregate()`` after
    all patients have been processed.  Subclasses accumulate whatever
    state is appropriate for their task type (segmentation, detection,
    classification, etc.) and return a plain dict of metrics from
    ``aggregate()``.

    Implementing a new evaluator
    ----------------------------
    1. Subclass ``BaseEvaluator``.
    2. Implement ``update(pred, target)`` to accumulate per-sample state.
    3. Implement ``aggregate()`` to return a flat ``dict[str, float|int]``
       of metrics that will be merged into ``MedicalImageModelTestingResults.metrics``.
    4. Pass an instance to ``ModelTestingPipeline(evaluator=...)``.

    The ``update`` contract
    -----------------------
    ``pred`` is the raw model output tensor for a single sample (no
    batch dimension assumed, but subclasses may handle either).
    ``target`` is the corresponding ground-truth tensor.  Both are
    already on the correct device when ``update`` is called.
    """

    @abstractmethod
    def update(self, pred, target) -> None:
        """Accumulate statistics for one sample."""

    @abstractmethod
    def aggregate(self) -> dict:
        """
        Return a flat dict of metrics computed over all accumulated
        samples.  Called once after the full test partition has been
        processed.
        """

    def reset(self) -> None:
        """
        Reset internal accumulators.  Called by the pipeline before
        each run so that an evaluator instance can be safely reused.
        Subclasses should override this if they hold mutable state.
        """