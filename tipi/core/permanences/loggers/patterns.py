"""Metric patterns for reproducible logging across backends."""

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

MetricStepPolicy = Literal["advance", "reuse_last"]


@dataclass(frozen=True)
class MetricRecord:
    """A single metric emission with a stable plotting pattern.

    Attributes:
        name: Unique metric identifier
        value: Metric value (scalar or structured)
        pattern: Plotting pattern category (e.g., "loss", "accuracy")
        stage: Training stage (e.g., "batch", "epoch", "evaluation")
        split: Data split (e.g., "train", "validation", "test")
        step: Explicit step override; if None, uses global step
        step_policy: How to handle step assignment:
            - "advance": increment global step after logging
            - "reuse_last": use the last logged step (for eval metrics)
    """

    name: str
    value: Any
    pattern: str
    stage: str | None = None
    split: str | None = None
    step: int | None = None
    step_policy: MetricStepPolicy = "advance"


@dataclass(frozen=True)
class ResolvedMetricRecord:
    """MetricRecord with a concrete resolved step."""

    metric: MetricRecord
    step: int

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def value(self) -> Any:
        return self.metric.value

    @property
    def pattern(self) -> str:
        return self.metric.pattern

    @property
    def stage(self) -> str | None:
        return self.metric.stage

    @property
    def split(self) -> str | None:
        return self.metric.split

    @property
    def step_policy(self) -> MetricStepPolicy:
        return self.metric.step_policy


@dataclass(frozen=True)
class MetricPattern:
    """Factory for reproducible metric records across logger backends.

    Enables users to create consistent MetricRecord instances via a pattern template.
    """

    name: str
    pattern: str
    stage: str | None = None
    split: str | None = None
    step_policy: MetricStepPolicy = "advance"

    def metric(self, value: Any, *, step: int | None = None) -> MetricRecord:
        """Create a metric record from this pattern."""
        return MetricRecord(
            name=self.name,
            value=value,
            pattern=self.pattern,
            stage=self.stage,
            split=self.split,
            step=step,
            step_policy=self.step_policy,
        )


@dataclass(frozen=True)
class MetricFigurePattern:
    """Definition for rebuilding a standard metric figure from logged records.

    Enables backends to reconstruct loss/accuracy curves from metric history.
    """

    name: str
    pattern: str
    title: str
    ylabel: str
    metric_names: tuple[str, ...] = ()

    def matches(self, metric_name: str, metric_pattern: str | None) -> bool:
        """Check if a metric belongs to this figure pattern."""
        if metric_pattern == self.pattern:
            return True
        return metric_name in self.metric_names


# Pre-defined metric patterns for common use cases
BATCH_LOSS = MetricPattern(
    name="batch_loss",
    pattern="loss",
    stage="batch",
    split="train",
)
EPOCH_LOSS = MetricPattern(
    name="epoch_loss",
    pattern="loss",
    stage="epoch",
    split="train",
)
VALIDATION_LOSS = MetricPattern(
    name="validation_loss",
    pattern="loss",
    stage="evaluation",
    split="validation",
    step_policy="reuse_last",
)
TEST_LOSS = MetricPattern(
    name="test_loss",
    pattern="loss",
    stage="evaluation",
    split="test",
    step_policy="reuse_last",
)
TRAIN_ACCURACY = MetricPattern(
    name="train_accuracy",
    pattern="accuracy",
    stage="epoch",
    split="train",
)
VALIDATION_ACCURACY = MetricPattern(
    name="validation_accuracy",
    pattern="accuracy",
    stage="evaluation",
    split="validation",
    step_policy="reuse_last",
)
TEST_ACCURACY = MetricPattern(
    name="test_accuracy",
    pattern="accuracy",
    stage="evaluation",
    split="test",
    step_policy="reuse_last",
)

# Pre-defined figure patterns for standard plots
LOSS_CURVE = MetricFigurePattern(
    name="loss",
    pattern="loss",
    title="Loss",
    ylabel="Loss",
    metric_names=("batch_loss", "epoch_loss", "validation_loss", "test_loss"),
)
ACCURACY_CURVE = MetricFigurePattern(
    name="accuracy",
    pattern="accuracy",
    title="Accuracy",
    ylabel="Accuracy",
    metric_names=("train_accuracy", "validation_accuracy", "test_accuracy"),
)


# Convenience functions for creating metric records
def batch_loss(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log batch loss metric."""
    return BATCH_LOSS.metric(value, step=step)


def epoch_loss(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log epoch loss metric."""
    return EPOCH_LOSS.metric(value, step=step)


def validation_loss(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log validation loss metric (reuses last training step)."""
    return VALIDATION_LOSS.metric(value, step=step)


def test_loss(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log test loss metric (reuses last training step)."""
    return TEST_LOSS.metric(value, step=step)


test_loss.__test__ = False  # Prevent pytest from treating this as a test


def train_accuracy(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log training accuracy metric."""
    return TRAIN_ACCURACY.metric(value, step=step)


def validation_accuracy(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log validation accuracy metric (reuses last training step)."""
    return VALIDATION_ACCURACY.metric(value, step=step)


def test_accuracy(value: Any, *, step: int | None = None) -> MetricRecord:
    """Log test accuracy metric (reuses last training step)."""
    return TEST_ACCURACY.metric(value, step=step)


test_accuracy.__test__ = False  # Prevent pytest from treating this as a test


MetricLogInput: TypeAlias = dict[str, Any] | MetricRecord | list[MetricRecord]
