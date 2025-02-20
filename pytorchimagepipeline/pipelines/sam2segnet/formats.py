import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PascalVocFormat:
    root: Path
    mean_std: dict = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    ignore = 255

    def __post_init__(self):
        with (self.root / "mean_std.json").open() as file_obj:
            self.mean_std = json.load(file_obj)

        with (self.root / "classes.json").open() as file_obj:
            self.classes = json.load(file_obj)

    def __len__(self):
        return len(self.data)

    def get_data(self, mode):
        data_file = self.root / f"ImageSets/Segmentation/{mode}.txt"

        if data_file.exists():
            with data_file.open() as file_obj:
                self.data = file_obj.read().split("\n")
        else:
            self.data = []
