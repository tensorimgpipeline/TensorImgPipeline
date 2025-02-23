import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(kw_only=True)
class DatasetFormat(ABC):
    data: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def get_data(self, mode: str) -> None: ...


@dataclass
class PascalVocFormat(DatasetFormat):
    root: Path | str
    mean_std: dict = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    ignore: int = 255

    def __post_init__(self) -> None:
        if isinstance(self.root, str):
            self.root = Path(self.root)
        with (self.root / "mean_std.json").open() as file_obj:
            self.mean_std = json.load(file_obj)

        with (self.root / "classes.json").open() as file_obj:
            self.classes = json.load(file_obj)

    def get_data(self, mode: str) -> None:
        if isinstance(self.root, str):
            self.root = Path(self.root)
        data_file = self.root / f"ImageSets/Segmentation/{mode}.txt"

        if data_file.exists():
            with data_file.open() as file_obj:
                self.data = file_obj.read().split("\n")
        else:
            self.data = []
