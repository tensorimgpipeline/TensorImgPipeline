from .permanences import Device, ProgressManager, WandBLogger
from .processes import ResultProcess

permanences_to_register = {
    "Device": Device,
    "ProgressManager": ProgressManager,
    "WandBLogger": WandBLogger,
}
processes_to_register = {
    "ResultProcess": ResultProcess,
}
