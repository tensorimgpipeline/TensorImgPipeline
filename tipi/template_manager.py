"""Template management for project scaffolding.

This module provides utilities for rendering Jinja2 templates for
creating new pipeline projects.
"""

from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, field_validator

# Define the allowed values as constants
# Using Literal for license_type can be more efficient and clear for Pydantic
ALLOWED_LICENSES = Literal["MIT", "GPLv3", "Apache-2.0"]
ALLOWED_EXAMPLE_NAMES = Literal["basic", "full", "pause"]


# based on https://www.datacamp.com/tutorial/python-user-input
class ProjectSetup(BaseModel):
    """
    Model to validate inputs for project setup function.

    Args:
        name: Name of the project
        base_dir: Base directory where project will be created
        example: Name of the example.
            allowed examples:
                - MIT
                - GPLv3
                - Apache-2.0
        description: Project description
        license_type: License type
            allowed license types:
                - basic
                - full
                - pause
    """

    name: str
    base_dir: Path
    example: ALLOWED_EXAMPLE_NAMES
    description: str | None = None
    # Literal handles the validation automatically
    license_type: ALLOWED_LICENSES = "MIT"

    # Validate 'base_dir' existence
    @field_validator("base_dir")
    @classmethod
    def check_base_dir_exists(cls, v: Path) -> Path:
        """Ensures the base_dir Path exists on the filesystem."""
        if not v.parent.is_dir():
            # In V2, the error message often doesn't need to include 'v'
            # as Pydantic handles context better, but it's safe to keep.
            raise ValueError(f"Parent of base directory does not exist or is not a directory: {v}")  # noqa: TRY003
        return v


class TemplateManager:
    """Manages Jinja2 templates for project scaffolding."""

    def __init__(self) -> None:
        """Initialize the template manager."""
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template file (relative to templates dir)
            context: Dictionary of variables to pass to the template

        Returns:
            Rendered template as a string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def create_project(self, project_data: ProjectSetup) -> None:
        """Create a complete project from templates.

        Args:
            project_data (ProjectSetup): Data Container for project creation.
                If the Datacontainer is used input validation is performed.
        """
        if project_data.description is None:
            project_data.description = f"A TensorImgPipeline project for {project_data.name}"

        context = {
            "project_name": project_data.name,
            "example": project_data.example,
            "description": project_data.description,
            "license": project_data.license_type,
        }

        # Create project structure
        src_dir = project_data.base_dir / project_data.name
        src_dir.mkdir(parents=True, exist_ok=True)
        config_dir = project_data.base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Render and write files
        self._write_file(src_dir / "__init__.py", "project/__init__.py.j2", context)
        self._write_file(src_dir / "permanences.py", "project/permanences.py.j2", context)
        self._write_file(src_dir / "processes.py", "project/processes.py.j2", context)
        self._write_file(config_dir / "pipeline_config.toml", "project/pipeline_config.toml.j2", context)
        self._write_file(project_data.base_dir / "README.md", "project/README.md.j2", context)
        self._write_file(project_data.base_dir / "pyproject.toml", "project/pyproject.toml.j2", context)
        self._write_file(project_data.base_dir / ".gitignore", "project/gitignore.j2", context)

    def _write_file(self, file_path: Path, template_name: str, context: dict[str, Any]) -> None:
        """Write a rendered template to a file.

        Args:
            file_path: Path where the file should be written
            template_name: Name of the template to render
            context: Template context
        """
        content = self.render_template(template_name, context)
        file_path.write_text(content)


# Global template manager instance
template_manager = TemplateManager()
