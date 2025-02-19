from .permanences import Device, ProgressManager, WandBLogger

permanences_to_register = {
    "Device": Device,
    "ProgressManager": ProgressManager,
    "WandBLogger": WandBLogger,
}
}
processes_to_register = {}
