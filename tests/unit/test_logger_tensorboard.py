import pytest

from tipi.core.permanences.loggers.patterns import ACCURACY_CURVE, LOSS_CURVE
from tipi.core.permanences.loggers.tensorboard import TensorBoardLogger


@pytest.mark.parametrize(
    "patterns, expected_layout",
    (
        (
            [LOSS_CURVE, ACCURACY_CURVE],
            {
                "Metrics": {
                    "Loss": [
                        "Multiline",
                        ["loss/batch_loss", "loss/epoch_loss", "loss/validation_loss", "loss/test_loss"],
                    ],
                    "Accuracy": [
                        "Multiline",
                        ["accuracy/train_accuracy", "accuracy/validation_accuracy", "accuracy/test_accuracy"],
                    ],
                }
            },
        ),
        (
            [LOSS_CURVE],
            {
                "Metrics": {
                    "Loss": [
                        "Multiline",
                        ["loss/batch_loss", "loss/epoch_loss", "loss/validation_loss", "loss/test_loss"],
                    ],
                }
            },
        ),
    ),
    ids=["both_curves", "loss_curve"],
)
def test_build_layout(patterns, expected_layout):
    logger = TensorBoardLogger(patterns=patterns)
    layout = logger._build_layout()

    assert layout == expected_layout
