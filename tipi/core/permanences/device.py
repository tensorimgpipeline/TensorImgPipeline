from __future__ import annotations

from dataclasses import dataclass

import torch

from tipi.abstractions import Permanence

__all__ = [
    "Device",
    "DeviceWithVRAM",
    "VRAMUsageError",
]


class VRAMUsageError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("All devices are using more than 80% of VRAM")


@dataclass
class DeviceWithVRAM:
    device: torch.device
    vram_usage: float


class Device(Permanence):
    """
    Device class for managing CUDA device selection based on VRAM usage.

    This class inherits from the Permanence class and is responsible for calculating
    and setting the best available CUDA device for computation based on VRAM usage.

    Example TOML Config:
        ```toml
        [permanences.network]
        type = "Network"
        params = { model = "deeplabv3_resnet50", num_classes = 21, pretrained = true }
        ```

    Methods:
        __init__() -> None:

        _calculate_best_device() -> None:
            Raises VRAMUsageError if the VRAM usage of the best device exceeds 80%.

        cleanup() -> None:
            Cleans up the device instance. This method doesn't perform any cleanup operations.
    """

    def __init__(self) -> None:
        """
        Initializes the instance and calculates the best device for computation.
        """
        devices = self._gather_device_statitics()
        self.device = self._calculate_best_device(devices)

    def is_initialized(self) -> bool:
        """Check if the device is initialized.

        Overrides the base method to provide logic for determining if the device is ready for use.

        Returns:
            bool: True if the device is initialized, False otherwise.
        """
        allowed_device_types = {
            "cuda": torch.cuda,
            "xpu": torch.xpu,
        }
        if self.device.type == "cpu":
            return True
        if self.device.type not in allowed_device_types:
            return False
        device_module = allowed_device_types.get(self.device.type)
        if device_module:
            return bool(device_module.is_available()) and bool(device_module.is_initialized())
        return False

    def _gather_device_statitics(self) -> list[DeviceWithVRAM]:
        """Gather device statistics using the official supported Accelerator stats.

        Returns:
            list[DeviceWithVRAM]: list of DeviceWithVRAM objects containing device and VRAM usage information.
        """
        supported_accelerators = [
            torch.cuda,
            torch.xpu,
            # torch.mtia, ## Planned but not implemented yet due to lack of access to hardware
            # torch.mps, ## Planned but not implemented yet due to lack of access to hardware
        ]

        devices = []

        for backend in supported_accelerators:
            if backend.is_available():
                backend_name = backend.__name__.split(".")[-1]
                backend_devices = [torch.device(f"{backend_name}:{i}") for i in range(backend.device_count())]
                devices.extend([
                    DeviceWithVRAM(
                        device,
                        backend.memory_reserved(device) / backend.get_device_properties(device).total_memory,
                    )
                    for device in backend_devices
                ])
        if not devices:
            devices.append(DeviceWithVRAM(torch.device("cpu"), 0.0))
        return devices

    def _calculate_best_device(self, devices: list[DeviceWithVRAM]) -> torch.device:
        """
        Calculate and set the best available CUDA device based on VRAM usage.

        This method iterates over all available CUDA devices, calculates their VRAM usage,
        and selects the device with the lowest VRAM usage. If the VRAM usage of the best
        device exceeds 80%, a VRAMUsageError is raised.

        Raises:
            VRAMUsageError: If the VRAM usage of the best device exceeds 80%.

        Attributes:
            device (torch.device): The CUDA device with the lowest VRAM usage.
        """
        best_device = min(devices, key=lambda x: x.vram_usage)

        if best_device.vram_usage > 0.8:
            raise VRAMUsageError()

        return best_device.device

    def cleanup(self) -> None:
        """
        Cleans up the device instance.

        This method doesn't perform any cleanup operations for the device instance.

        Returns:
            None: This method doesn't return any value.
        """
        pass
