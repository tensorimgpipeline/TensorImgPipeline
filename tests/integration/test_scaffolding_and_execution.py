import pytest

from tests.conftest import skip_outside_container
from tipi.core.runner import PipelineRunner
from tipi.template_manager import ProjectSetup, template_manager

# Apply skip_outside_container to all tests in this module
pytestmark = skip_outside_container


class TestExampleScaffold:
    """Test template_manager to create scaffold project."""

    @pytest.mark.parametrize(
        argnames="example, desc, exp_out, exp_err",
        argvalues=(
            (
                "full",
                "Test Full",
                (
                    5,
                    [
                        "Loading data from data",
                        "Loaded 100 samples",
                        "Processing data with batch size 32",
                        "Processed 100 samples into 3 batches",
                        "",
                    ],
                ),
                (1, [""]),
            ),
            ("basic", "Test Basic", (2, ["Executing MyProcess", ""]), (1, [""])),
        ),
        ids=("full", "basic"),
    )
    def test_scaffold_working_project(self, isolated_path_manager, capsys, example, desc, exp_out, exp_err):
        """Test the default basic scaffolding and execute it."""
        path_mgr, dirs = isolated_path_manager
        project_name = "test_basic_project"

        # Create project in a workspace directory (simulating user's project location)
        workspace = dirs.root / "workspace"
        workspace.mkdir()
        project_dir = workspace / project_name

        setup = ProjectSetup(name=project_name, base_dir=project_dir, example=example, description=desc)

        template_manager.create_project(setup)

        config = project_dir / "configs" / "pipeline_config.toml"
        project_package = project_dir / project_name
        readme = project_dir / "README.md"

        # Check structure
        assert project_package.exists()
        assert readme.exists()
        assert config.exists()

        # Link the project to the isolated projects directory (simulating `tipi add`)
        link_path = dirs.projects / project_name
        link_path.symlink_to(project_package)

        # Link configs as well
        config_link = dirs.configs / project_name
        config_link.symlink_to(project_dir / "configs")

        # Verify we're using the test directory, not ~/.config
        assert path_mgr.get_projects_dir() == dirs.projects
        assert path_mgr.get_configs_dir() == dirs.configs

        runner = PipelineRunner(project_name, config)
        runner.run()

        captured = capsys.readouterr()

        stdout_lines = captured.out.split("\n")
        stderr_lines = captured.err.split("\n")

        assert len(stdout_lines) == exp_out[0]
        assert stdout_lines == exp_out[1]
        assert len(stderr_lines) == exp_err[0]
        assert stderr_lines == exp_err[1]
