from dataclasses import dataclass, field

from pytorchimagepipeline.abstractions import AbstractConfig
from pytorchimagepipeline.errors import InvalidConfigError


@dataclass
class WandBLoggerConfig(AbstractConfig):
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
