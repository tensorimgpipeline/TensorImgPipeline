import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from tipi.abstractions import AbstractConfig
from tipi.errors import InvalidConfigError

T = TypeVar("T", bound=BaseModel)


class ConfigReader(Protocol):
    def read(self, filepath: Path) -> dict[str, Any]: ...


class JSONReader:
    def read(self, filepath: Path) -> dict[str, Any]:
        with filepath.open("r", encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))


class TOMLReader:
    def read(self, filepath: Path) -> dict[str, Any]:
        with filepath.open("rb") as f:
            return tomllib.load(f)


class ConfigLoader:
    def __init__(self) -> None:
        self._readers: dict[str, ConfigReader] = {}

    def register_reader(self, extension: str, reader: ConfigReader) -> None:
        self._readers[extension.lower().lstrip(".")] = reader

    def load(self, filepath: str | Path) -> dict[str, Any]:
        """Load config file based on its extension using the registered readers."""
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        reader = self._readers.get(ext)
        if not reader:
            raise ValueError(f"Kein Reader für '.{ext}' registriert.")
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")
        return reader.read(path)

    def load_as(self, filepath: str | Path, model: type[T]) -> T:
        """Load Config via protocol composition reader and validate it with pydantic model."""
        raw_data = self.load(filepath)
        return model.model_validate(raw_data)


config_loader = ConfigLoader()
config_loader.register_reader("json", JSONReader())
config_loader.register_reader("toml", TOMLReader())


@dataclass
class WandManagerConfig(AbstractConfig):
    entity: str
    project: str
    name: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    count: int = 1

    def validate(self) -> None:
        # TODO How to querry projects, entity, etc from wandb
        if not isinstance(self.entity, str):
            raise InvalidConfigError(context="wrong-type-entity", value=self.entity)
        if not isinstance(self.project, str):
            raise InvalidConfigError(context="wrong-type-project", value=self.project)
        if not isinstance(self.name, str):
            raise InvalidConfigError(context="wrong-type-run-name", value=self.name)
        if self.tags and any(type(y) is not str for y in self.tags):
            raise InvalidConfigError(context="wrong-type-tags", value="|".join(self.tags))
