"""End-to-end tests for TensorImgPipeline using fake filesystem."""

import re
from unittest.mock import patch

import pytest
from git import Repo

from tests.conftest import skip_no_network, skip_outside_container
from tipi.cli import app

# Apply skip_outside_container to all tests in this module
pytestmark = skip_outside_container


class TestE2ECreate:
    """End-to-end tests for create command with fake filesystem."""

    @pytest.mark.order(1)
    @pytest.mark.usefixtures("isolated_path_manager")
    @pytest.mark.parametrize(
        argnames=("project_name", "cliargs", "expected"),
        argvalues=[
            # Expected Values
            #   (ExitCode, Output Message, # Lines Output)
            ("DemoPipeline", ["--example", "basic"], (0, "Project Created", 30)),
            ("Demo Pipeline", ["--example", "basic"], (0, "Project Created", 30)),
            ("DemoPipeline", ["--example", "full"], (0, "Project Created", 30)),
        ],
        ids=("simple_basic", "simple_w_spaces", "simple_full"),
    )
    def test_create_new_pipeline(self, tmp_path, cli_runner, project_name, cliargs, expected):
        """Test creating a new pipeline project."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create in workspace
        cmd = ["create", project_name, "-l", str(workspace)]
        cmd.extend(cliargs)
        result = cli_runner.invoke(app, cmd)

        assert result.exit_code == expected[0]
        assert expected[1] in result.output
        assert len(result.output.split("\n")) == expected[2]

        # Analyes created project
        project, *rest = workspace.glob("*")
        assert len(rest) == 0
        assert project.name == project_name


class TestE2EAdd:
    """End-to-end tests for add command with fake filesystem."""

    @pytest.mark.order(2)
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_add_pipeline(self, tmp_path, cli_runner):
        """Test adding a pipeline to configuration."""
        # Use real filesystem for this test since inspect needs to import modules
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        result = cli_runner.invoke(app, ["add", str(workspace / "DemoProject")])

        assert result.exit_code == 0
        assert result.stderr == ""
        assert len(result.stdout.split("\n")) == 16
        assert "DemoProject" in result.stdout

        # Verify symlinks were created (should be in fake home due to env overrides)
        assert (tmp_path / "projects/DemoProject").exists()
        assert (tmp_path / "projects/DemoProject").is_symlink()

    @pytest.mark.parametrize(
        argnames=("cliargs", "expected"),
        argvalues=(
            # Expected format: ProjectName, Location, BranchName
            ([], ["DemoPipeline", "cache/projects/DemoPipeline", "main"]),
            (["-l"], ["DemoPipeline", "workspace/DemoPipeline", "main"]),
            (["-b", "develop"], ["DemoPipeline", "cache/projects/DemoPipeline", "develop"]),
        ),
        ids=("NoArgs", "Location", "Branch"),
    )
    @pytest.mark.order(3)
    @skip_no_network
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_add_pipeline_remote(self, tmp_path, cli_runner, cliargs, expected):
        """Test adding a pipeline to configuration."""
        # Use real filesystem for this test since inspect needs to import modules
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        project_url = "https://github.com/tensorimgpipeline/DemoPipeline.git"
        cmd = ["add", project_url]
        if "-l" in cliargs:
            cliargs.append(str(workspace))
        cmd.extend(cliargs)
        result = cli_runner.invoke(app, cmd)

        assert result.exit_code == 0
        assert result.stderr == ""
        assert len(result.stdout.split("\n")) == 23
        assert expected[0] in result.stdout

        # Verify symlinks were created (should be in fake home due to env overrides)
        assert (tmp_path / "projects/DemoPipeline").exists()
        assert (tmp_path / "projects/DemoPipeline").is_symlink()
        assert (tmp_path / expected[1]).is_dir()
        assert Repo(path=tmp_path / expected[1]).active_branch.name == expected[2]

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_add_pipeline_failing(self, tmp_path, cli_runner):
        """Test that add command links both code and configs."""
        # Use real filesystem for this test since inspect needs to import modules
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        result = cli_runner.invoke(app, ["add", "DemoProject"])

        assert result.exit_code == 1


class TestE2EInfo:
    """End-to-end tests for info command."""

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_info_command_real_filesystem(self, cli_runner, tmp_path):
        """Test info command with real filesystem."""

        result = cli_runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "TensorImgPipeline Configuration" in result.stdout

        # Check paths use isolated temp directory
        assert str(tmp_path) in result.stdout

        # Check directory status indicators (all should exist due to isolated_path_manager)
        assert "✓" in result.stdout

        # Check table structure
        assert "Setting" in result.stdout
        assert "Value" in result.stdout
        assert "Directory Status" in result.stdout


class TestE2EList:
    """End-to-end tests for list command with fake filesystem."""

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_list_empty_projects(self, cli_runner):
        """Test list command with no projects."""

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 1
        assert "No pipelines found" in result.stderr

    @pytest.mark.parametrize(
        argnames=("cliargs", "exp_stdout"),
        argvalues=(
            (([], ["Source", "/tmp/pytest"])),
            ((["-v"], ["Permanenc", "Processes", "Source", "2", "2", "/tmp/pyte"])),
            ((["--no-links"], [])),
        ),
        ids=("simple", "simple-v", "simple--no-links"),
    )
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_list_with_project(self, tmp_path, cli_runner, cliargs, exp_stdout):
        """Test list command after adding a project."""

        default_expected_stdout = ["Pipeline", "Type", "Config", "Status", "DemoProject", "Linked", "✓", "Ready"]

        default_expected_stdout.extend(exp_stdout)

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / "DemoProject")])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        cmd = ["list"]
        cmd.extend(cliargs)
        result = cli_runner.invoke(app, cmd)

        # Should list pipelines if path resolution works
        assert result.exit_code == 0
        for value in default_expected_stdout:
            assert value in result.stdout


class TestE2ERemove:
    """End-to-end tests for remove command with fake filesystem."""

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_remove_pipeline(self, tmp_path, cli_runner):
        """Test removing a pipeline."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / "DemoProject")])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        project_path = tmp_path / "projects/DemoProject"
        config_path = tmp_path / "configs/DemoProject"

        # Verify symlinks exists before removing
        assert project_path.exists()
        assert config_path.exists()

        # Now remove it
        result = cli_runner.invoke(app, ["remove", "DemoProject"])

        # Check result
        assert result.exit_code == 0
        assert "✓ Removed link" in result.stdout
        assert "✓ Removed config link" in result.stdout
        assert "✓ Pipeline 'DemoProject' removed successfully!" in result.stdout

        # Verify symlinks were removed
        assert not project_path.exists()
        assert not config_path.exists()

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_remove_pipeline_remote_source(self, tmp_path, cli_runner):
        """Test removing pipeline which has remote backup"""
        # Add Pipeline via remote url
        project_url = "https://github.com/tensorimgpipeline/DemoPipeline.git"
        cmd = ["add", project_url]
        create_result = cli_runner.invoke(app, cmd)
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        project_path = tmp_path / "projects/DemoPipeline"
        config_path = tmp_path / "configs/DemoPipeline"
        cached_project = tmp_path / "cache/projects/DemoPipeline"

        # Verify symlinks exists before removing
        assert project_path.exists()
        assert config_path.exists()
        assert cached_project.exists()

        # Now remove it
        result = cli_runner.invoke(app, ["remove", "DemoPipeline", "--delete-source"], input="y\n")

        # Check result
        assert result.exit_code == 0
        assert "✓ Removed link" in result.stdout
        assert "✓ Removed config link" in result.stdout
        assert "✓ Deleted source" in result.stdout
        assert "✓ Pipeline 'DemoPipeline' removed successfully!" in result.stdout

        # Verify symlinks were removed
        assert not project_path.exists()
        assert not config_path.exists()
        assert not cached_project.exists()

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_remove_pipeline_source(self, tmp_path, cli_runner):
        """Test removing a pipeline."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / "DemoProject")])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        project_path = tmp_path / "projects/DemoProject"
        config_path = tmp_path / "configs/DemoProject"

        # Verify symlinks exists before removing
        assert project_path.exists()
        assert config_path.exists()

        # Now remove it
        result = cli_runner.invoke(app, ["remove", "DemoProject", "--delete-source"], input="y\n")

        # Check result
        assert result.exit_code == 0
        assert "✓ Removed link" in result.stdout
        assert "✓ Removed config link" in result.stdout
        assert "✓ Pipeline 'DemoProject' removed successfully!" in result.stdout
        assert "Warning: Source is outside cache/, not deleting for safety." in result.stdout

        # Verify symlinks were removed
        assert not project_path.exists()
        assert not config_path.exists()

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_remove_nonexistent_pipeline(self, cli_runner):
        """Test removing a pipeline that doesn't exist."""

        result = cli_runner.invoke(app, ["remove", "nonexistent"])

        assert result.exit_code != 0


class TestE2EInspect:
    """End-to-end tests for inspect command.

    Note: These tests use real filesystem (tmp_path) instead of fakefs because
    the inspect command needs to import Python modules, which requires real files
    that importlib can access.
    """

    @pytest.mark.parametrize(
        argnames=("cmd", "expected"),
        argvalues=(
            # expected structure: (ErrorCode, # Lines in Output, Regex with expected matches)
            (["inspect", "DemoProject"], (0, 12, {})),
            (["inspect", "DemoProject", "-d"], (0, 16, {4: r":\n^[│\s](\s{3}│\s{5})|(\s{9,11})"})),
            (["inspect", "DemoProject", "--docs"], (0, 16, {4: r":\n^[│\s](\s{3}│\s{5})|(\s{9,11})"})),
        ),
        ids=("simple", "simple-d", "simple--docs"),
    )
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_inspect_pipeline(self, cli_runner, tmp_path, cmd, expected):
        """Test inspecting a pipeline shows permanences and processes."""
        # Use real filesystem for this test since inspect needs to import modules
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a project with the full example template
        create_result = cli_runner.invoke(app, ["create", "DemoProject", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / "DemoProject")])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        # Inspect the project
        result = cli_runner.invoke(app, cmd)

        assert result.exit_code == expected[0], f"Inspect failed: {result.output}"
        assert len(result.output.split("\n")) == expected[1]
        assert cmd[1] in result.output
        if expected[2]:
            for num_exp, pattern in expected[2].items():
                matches = re.finditer(pattern, result.output, re.MULTILINE)
                assert len(list(matches)) == num_exp
        # Full example should have permanences and processes defined
        assert "DataPermanence" in result.output
        assert "ConfigPermanence" in result.output
        assert "ProcessDataProcess" in result.output
        assert "LoadDataProcess" in result.output

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_inspect_nonexistent_pipeline(self, cli_runner):
        """Test inspecting a pipeline that doesn't exist."""
        result = cli_runner.invoke(app, ["inspect", "NonExistentProject"])

        assert result.exit_code == 1
        assert "Error: Pipeline 'NonExistentProject' not found." in result.stderr


class TestE2EWorkflow:
    """End-to-end workflow tests combining multiple commands."""

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_full_pipeline_lifecycle(self, cli_runner, tmp_path):
        """Test complete pipeline lifecycle: create -> add -> list -> remove -> add -> run."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # 1. Create Demo Pipeline as full example
        create_result = cli_runner.invoke(app, ["create", "demo_pipeline", "-l", str(workspace), "-e", "full"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        demo_path = workspace / "demo_pipeline"
        assert demo_path.exists(), "Demo pipeline directory was not created"

        # 2. Add pipeline
        add_result = cli_runner.invoke(app, ["add", str(demo_path)])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        # 3. List pipelines (should show demo_pipeline)
        list_result = cli_runner.invoke(app, ["list"])
        assert list_result.exit_code == 0, f"List failed: {list_result.output}"
        assert "demo_pipeline" in list_result.stdout

        # 4. Remove pipeline
        remove_result = cli_runner.invoke(app, ["remove", "demo_pipeline"], input="y\n")
        assert remove_result.exit_code == 0, f"Remove failed: {remove_result.output}"

        # 5. List again (should not show demo_pipeline)
        list_result2 = cli_runner.invoke(app, ["list"])
        assert list_result2.exit_code == 1, f"List failed: {list_result2.stderr}"
        assert "demo_pipeline" not in list_result2.stderr, "Pipeline should be removed"

        # 6. Add pipeline again
        add_result2 = cli_runner.invoke(app, ["add", str(demo_path)])
        assert add_result2.exit_code == 0, f"Re-add failed: {add_result2.output}"

        # 7. Run pipeline
        run_result = cli_runner.invoke(app, ["run", "demo_pipeline"])
        assert run_result.exit_code == 0, f"Run crashed: {run_result.output}"


class TestE2ERun:
    """End-to-end tests for run command with output verification."""

    @pytest.mark.order(4)
    @pytest.mark.usefixtures("isolated_path_manager")
    @pytest.mark.parametrize(
        argnames=("example", "expected_output"),
        argvalues=[
            ("basic", ["Executing MyProcess"]),
            ("full", ["Loading data from", "Loaded", "samples", "Processing data"]),
            ("pause", []),  # pause template has progress bars but no text output
        ],
        ids=("basic", "full", "pause"),
    )
    def test_run_pipeline_success(self, cli_runner, tmp_path, example, expected_output):
        """Test running a pipeline completes successfully and produces expected output."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        project_name = f"Demo{example.capitalize()}"

        # 1. Create project with specified example template
        create_result = cli_runner.invoke(app, ["create", project_name, "-l", str(workspace), "-e", example])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # 2. Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / project_name)])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        # 3. Run the pipeline
        run_result = cli_runner.invoke(app, ["run", project_name])
        assert run_result.exit_code == 0, f"Run failed: {run_result.output}"

        # 4. Verify expected output is present
        stdout = run_result.stdout
        for expected in expected_output:
            assert expected in stdout, f"Expected '{expected}' not found in output:\n{stdout}"

    @pytest.mark.order(5)
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_run_pipeline_with_progress_bar(self, cli_runner, tmp_path):
        """Test running a pipeline with enable_progress=true shows progress bars.

        The 'pause' template has enable_progress=true in pipeline_config.toml,
        which enables the ProgressManager and shows progress bars for
        'overall' (processes) and 'cleanup' (permanences) phases.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        project_name = "DemoPause"

        # Create project with pause example (has enable_progress=true)
        create_result = cli_runner.invoke(app, ["create", project_name, "-l", str(workspace), "-e", "pause"])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        # Add the project
        add_result = cli_runner.invoke(app, ["add", str(workspace / project_name)])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        # Run the pipeline
        run_result = cli_runner.invoke(app, ["run", project_name])
        assert run_result.exit_code == 0, f"Run failed: {run_result.output}"

        stdout = run_result.stdout

        # Verify progress bar output for overall (1 process)
        # Pattern: "overall" followed by progress bar showing (1/1)
        overall_pattern = r"overall\s+.*?\(1/1\)"
        overall_match = re.search(overall_pattern, stdout)
        assert overall_match is not None, f"Expected overall progress (1/1) not found in output:\n{stdout}"

        # Verify progress bar output for cleanup
        # Note: Cleanup shows (2/2) because it includes MyPermanence + ProgressManager
        # Pattern: "cleanup" followed by progress bar showing (N/N) where N >= 1
        cleanup_pattern = r"cleanup\s+.*?\((\d+)/\1\)"
        cleanup_match = re.search(cleanup_pattern, stdout)
        assert cleanup_match is not None, f"Expected cleanup progress (N/N) not found in output:\n{stdout}"
        # Verify at least 2 permanences (MyPermanence + ProgressManager)
        cleanup_count = int(cleanup_match.group(1))
        assert cleanup_count >= 2, f"Expected cleanup count >= 2, got {cleanup_count}"

    @pytest.mark.order(6)
    @pytest.mark.usefixtures("isolated_path_manager")
    def test_run_nonexistent_pipeline(self, cli_runner):
        """Test running a pipeline that doesn't exist."""
        result = cli_runner.invoke(app, ["run", "NonExistentProject"])

        assert result.exit_code != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    @pytest.mark.order(7)
    @pytest.mark.usefixtures("isolated_path_manager")
    @pytest.mark.parametrize(
        argnames="example",
        argvalues=["basic", "full", "pause"],
        ids=("basic", "full", "pause"),
    )
    def test_run_pipeline_twice_succeeds(self, cli_runner, tmp_path, example):
        """Test that running a pipeline twice succeeds (idempotent execution)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        project_name = f"Test{example.capitalize()}"

        # Create and add
        create_result = cli_runner.invoke(app, ["create", project_name, "-l", str(workspace), "-e", example])
        assert create_result.exit_code == 0, f"Create failed: {create_result.output}"

        add_result = cli_runner.invoke(app, ["add", str(workspace / project_name)])
        assert add_result.exit_code == 0, f"Add failed: {add_result.output}"

        # Run first time
        run_result1 = cli_runner.invoke(app, ["run", project_name])
        assert run_result1.exit_code == 0, f"First run failed: {run_result1.output}"

        # Run second time
        run_result2 = cli_runner.invoke(app, ["run", project_name])
        assert run_result2.exit_code == 0, f"Second run failed: {run_result2.output}"


class TestE2EEnvironmentOverrides:
    """Test environment variable overrides with fake filesystem."""

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_custom_projects_dir(self, tmp_path, cli_runner):
        """Test TIPI_PROJECTS_DIR override."""

        custom_projects = tmp_path / "custom/projects"

        with patch.dict("os.environ", {"TIPI_PROJECTS_DIR": str(custom_projects)}):
            result = cli_runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert str(custom_projects) in result.stdout

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_custom_config_dir(self, tmp_path, cli_runner):
        """Test TIPI_CONFIG_DIR override."""

        custom_config = tmp_path / "custom/configs"

        with patch.dict("os.environ", {"TIPI_CONFIG_DIR": str(custom_config)}):
            result = cli_runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert str(custom_config) in result.stdout

    @pytest.mark.usefixtures("isolated_path_manager")
    def test_custom_cache_dir(self, tmp_path, cli_runner):
        """Test TIPI_CACHE_DIR override."""

        custom_cache = tmp_path / "custom/cache"

        with patch.dict("os.environ", {"TIPI_CACHE_DIR": str(custom_cache)}):
            result = cli_runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert str(custom_cache) in result.stdout
