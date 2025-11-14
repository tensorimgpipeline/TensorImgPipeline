"""Template management for project scaffolding.

This module provides utilities for rendering Jinja2 templates for
creating new pipeline projects.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateManager:
    """Manages Jinja2 templates for project scaffolding."""

    def __init__(self):
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

    def create_project(
        self,
        project_name: str,
        base_dir: Path,
        with_example: bool = False,
        description: str | None = None,
        license_type: str = "MIT",
    ) -> None:
        """Create a complete project from templates.

        Args:
            project_name: Name of the project
            base_dir: Base directory where project will be created
            with_example: Whether to include example code
            description: Project description
            license_type: License type (e.g., "MIT", "Apache-2.0")
        """
        if description is None:
            description = f"A PytorchImagePipeline project for {project_name}"

        context = {
            "project_name": project_name,
            "with_example": with_example,
            "description": description,
            "license": license_type,
        }

        # Create project structure
        src_dir = base_dir / project_name
        src_dir.mkdir(parents=True, exist_ok=True)
        config_dir = base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Render and write files
        self._write_file(src_dir / "__init__.py", "project/__init__.py.j2", context)
        self._write_file(src_dir / "permanences.py", "project/permanences.py.j2", context)
        self._write_file(src_dir / "processes.py", "project/processes.py.j2", context)
        self._write_file(config_dir / "pipeline_config.toml", "project/pipeline_config.toml.j2", context)
        self._write_file(base_dir / "README.md", "project/README.md.j2", context)
        self._write_file(base_dir / "pyproject.toml", "project/pyproject.toml.j2", context)
        self._write_file(base_dir / ".gitignore", "project/gitignore.j2", context)

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
