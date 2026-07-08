from tipi.core.permanences.loggers import patterns as _patterns
from tipi.core.permanences.loggers.basic import BasicLogger
from tipi.core.permanences.loggers.patterns import __all__ as _patterns_all
from tipi.core.permanences.loggers.tensorboard import TensorBoardLogger
from tipi.core.permanences.loggers.wandb import NullWandBLogger, WandBLogger

# Re-export patterns symbols declared by the patterns module contract.
for _name in _patterns_all:
    globals()[_name] = getattr(_patterns, _name)

__all__ = (
    "BasicLogger",
    "NullWandBLogger",
    "TensorBoardLogger",
    "WandBLogger",
    *_patterns_all,
)
