"""Abstract base classes for TensorImgPipeline.

This package provides the core abstractions used throughout the pipeline framework.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from tipi.abstractions.config import AbstractConfig, ProcessConfig
from tipi.abstractions.controller import AbstractController
from tipi.abstractions.permanence import Permanence
from tipi.abstractions.process import PipelineProcess

__all__ = [
    "AbstractConfig",
    "AbstractController",
    "Permanence",
    "PipelineProcess",
    "ProcessConfig",
]
