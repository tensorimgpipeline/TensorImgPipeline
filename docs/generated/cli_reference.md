# CLI Reference

Complete reference for the `pytorchpipeline` command-line tool.

## Overview

The `pytorchpipeline` CLI supports the complete workflow from creating new pipeline projects to running them in production.

```bash
pytorchpipeline [COMMAND] [OPTIONS]
```

## Commands

### `run` - Execute a Pipeline

Run a pipeline by name.

```bash
pytorchpipeline run PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to run (e.g., 'sam2segnet')

**Options:**

- `--config`, `-c` - Path to custom config file (relative to configs/)

**Examples:**

```bash
# Run default pipeline
pytorchpipeline run sam2segnet

# Run with custom config
pytorchpipeline run my_pipeline --config custom.toml
```

---

### `list` - List Available Pipelines

Show all available pipelines including linked subpackages.

```bash
pytorchpipeline list [OPTIONS]
```

**Options:**

- `--verbose`, `-v` - Show detailed information (permanences and processes count)
- `--links/--no-links` - Show/hide symlink source paths (default: show)

**Examples:**

```bash
# Simple list
pytorchpipeline list

# Detailed view
pytorchpipeline list -v

# Hide link sources
pytorchpipeline list --no-links
```

**Output:**

- Pipeline name
- Type (Built-in or Linked)
- Config status
- Status (Ready or Missing config)
- Permanences count (with `-v`)
- Processes count (with `-v`)
- Source path (for linked packages, with `--links`)

---

### `inspect` - Inspect Pipeline Components

Show detailed information about a pipeline's permanences and processes.

```bash
pytorchpipeline inspect PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to inspect

**Options:**

- `--docs`, `-d` - Show docstrings for components

**Examples:**

```bash
# Basic inspection
pytorchpipeline inspect sam2segnet

# With docstrings
pytorchpipeline inspect sam2segnet --docs
```

**Output:**

- Tree view of permanences and processes
- Class names and registration names
- Docstrings (with `--docs`)
- Config file location and status

---

### `create` - Create New Pipeline Project

Create a new standalone pipeline project with complete scaffolding.

```bash
pytorchpipeline create PROJECT_NAME [OPTIONS]
```

**Arguments:**

- `PROJECT_NAME` - Name for the new pipeline project

**Options:**

- `--location`, `-l` - Directory to create project in (default: current directory)
- `--example`, `-e` - Include example permanence and process implementations

**Examples:**

```bash
# Create in current directory
pytorchpipeline create my_pipeline

# Create in specific location
pytorchpipeline create my_pipeline --location ./projects

# Create with examples
pytorchpipeline create my_pipeline --example
```

**Creates:**

```
my_pipeline/
├── my_pipeline/
│   ├── __init__.py           # Component registration
│   ├── permanences.py        # Permanence implementations
│   └── processes.py          # Process implementations
├── configs/
│   └── pipeline_config.toml  # Configuration file
├── pyproject.toml            # Package metadata
├── README.md                 # Documentation
└── .gitignore               # Git ignore rules
```

---

### `add` - Link or Clone Pipeline Package

Link an existing local project or clone from a git repository as a subpackage.

```bash
pytorchpipeline add SOURCE [OPTIONS]
```

**Arguments:**

- `SOURCE` - Local path to project OR git repository URL

**Options:**

- `--name`, `-n` - Name for the linked pipeline (defaults to directory name)
- `--location`, `-l` - Location to clone git repos (default: ./submodules)
- `--branch`, `-b` - Git branch to checkout when cloning

**Examples:**

```bash
# Link local project
pytorchpipeline add ./my_pipeline_project
pytorchpipeline add /path/to/project --name custom_name

# Clone from git (HTTPS)
pytorchpipeline add https://github.com/user/ml-pipeline.git

# Clone from git (SSH)
pytorchpipeline add git@github.com:user/pipeline.git

# Clone specific branch
pytorchpipeline add https://github.com/user/repo.git --branch dev

# Clone to custom location
pytorchpipeline add https://github.com/user/pipeline.git --location ./external
```

**Behavior:**

- **Local path**: Creates a symlink in the projects directory (development: `pytorchimagepipeline/pipelines/`, production: `~/.config/pytorchimagepipeline/projects/`)
- **Git URL**: Clones to specified location (or cache directory), then creates symlink
- Validates package structure (must have `__init__.py`)
- Checks for naming conflicts
- Confirms before overwriting existing links

---

### `remove` - Remove Linked Package

Remove a linked subpackage from the pipeline system.

```bash
pytorchpipeline remove PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the linked pipeline to remove

**Options:**

- `--delete-source` - Also delete the source directory (only safe for cloned repos in submodules/)

**Examples:**

```bash
# Just remove the link
pytorchpipeline remove my_pipeline

# Remove link and delete source (if in submodules/)
pytorchpipeline remove my_pipeline --delete-source
```

**Safety:**

- Built-in pipelines cannot be removed
- Source deletion only works for packages in cache directory
- Confirms before deleting source directories

---

### `validate` - Validate Pipeline Configuration

Check that a pipeline is properly configured and ready to run.

```bash
pytorchpipeline validate PIPELINE_NAME
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to validate

**Examples:**

```bash
pytorchpipeline validate sam2segnet
```

**Checks:**

- ✓ Pipeline module exists and is importable
- ✓ Permanences and processes are registered
- ✓ Registered classes inherit from correct base classes
- ✓ Config file exists and is valid TOML
- ✓ Config has required sections (permanences/processes)

**Output:**

- Green panel if all checks pass
- Red panel with issues if validation fails

---

### `info` - Show Installation Information

Display information about the current PytorchImagePipeline installation and configuration.

```bash
pytorchpipeline info
```

**Examples:**

```bash
pytorchpipeline info
```

**Output:**

- Projects directory path (where pipelines are loaded from)
- Configs directory path (where configuration files are stored)
- Cache directory path (for cloned repositories)
- User config directory (~/.config/pytorchimagepipeline)
- Directory existence status

**Default Paths:**

- Projects: `~/.config/pytorchimagepipeline/projects/`
- Configs: `~/.config/pytorchimagepipeline/configs/`
- Cache: `~/.cache/pytorchimagepipeline/`

**Use Cases:**

- Debugging path issues
- Understanding where to place pipeline projects
- Checking directory structure before adding pipelines

---

## Complete Workflow Example

### Starting from Scratch

```bash
# 1. Create a new pipeline project
pytorchpipeline create my_ml_pipeline --example


# 2. Navigate to the project
cd my_ml_pipeline

# 3. Edit components (in your editor)
# - Edit my_ml_pipeline/permanences.py
# - Edit my_ml_pipeline/processes.py
# - Update configs/pipeline_config.toml

# 4. Go back to main pipeline directory
cd /path/to/PytorchPipeline

# 5. Link your project
pytorchpipeline add ../my_ml_pipeline

# 6. Validate configuration
pytorchpipeline validate my_ml_pipeline

# 7. Run the pipeline
pytorchpipeline run my_ml_pipeline
```

### Using an Existing Git Repository

```bash
# 1. Clone and link a pipeline from GitHub
pytorchpipeline add https://github.com/user/awesome-pipeline.git

# 2. Check what was added
pytorchpipeline list -v

# 3. Inspect the components
pytorchpipeline inspect awesome-pipeline --docs

# 4. Validate before running
pytorchpipeline validate awesome-pipeline

# 5. Run it
pytorchpipeline run awesome-pipeline
```

### Managing Multiple Pipelines

```bash
# List all pipelines
pytorchpipeline list -v

# Inspect each one
pytorchpipeline inspect pipeline1
pytorchpipeline inspect pipeline2

# Run specific ones
pytorchpipeline run pipeline1
pytorchpipeline run pipeline2 --config custom.toml

# Remove one that's no longer needed
pytorchpipeline remove old_pipeline
```

---

## Configuration Files

### Project Structure

When you create a pipeline with `create`, you get this structure:

```
my_pipeline/
├── my_pipeline/              # Python package
│   ├── __init__.py          # Registers components
│   ├── permanences.py       # Permanence classes
│   └── processes.py         # Process classes
├── configs/                 # Configuration directory
│   └── pipeline_config.toml # Pipeline configuration
├── pyproject.toml           # Package metadata
├── README.md               # Documentation
└── .gitignore             # Git ignore rules
```

### Configuration File Format

`configs/pipeline_config.toml`:

```toml
# Define permanences
[permanences.device]
type = "Device"  # Class name from __init__.py

[permanences.network]
type = "Network"
params = { model = "resnet50", num_classes = 10 }

# Define processes
[processes.training]
type = "TrainingProcess"
params = { epochs = 10, learning_rate = 0.001 }

[processes.validation]
type = "ValidationProcess"
```

---

## Environment Variables

- `PYTORCHPIPELINE_CONFIG_DIR` - Override configs directory
- `PYTORCHPIPELINE_PROJECTS_DIR` - Override projects directory
- `PYTORCHPIPELINE_CACHE_DIR` - Override cache directory

**Note:** Environment variables are primarily for testing and creating isolated test scenarios.

---

## Tips & Tricks

### Quick Inspection

```bash
# Combine list and inspect for quick overview
pytorchpipeline list -v && pytorchpipeline inspect sam2segnet
```

### Development Workflow

```bash
# During development, frequently validate
pytorchpipeline validate my_pipeline && pytorchpipeline run my_pipeline
```

### Working with Git Repos

```bash
# Clone, inspect, then run
pytorchpipeline add https://github.com/user/pipeline.git && \
  pytorchpipeline inspect pipeline --docs && \
  pytorchpipeline validate pipeline
```

### Batch Operations

```bash
# Validate all pipelines
for pipeline in $(pytorchpipeline list --no-links | tail -n +3 | awk '{print $1}'); do
    echo "Validating $pipeline..."
    pytorchpipeline validate $pipeline
done
```

---

## Troubleshooting

### "Pipeline not found"

- Check `pytorchpipeline list` to see available pipelines
- Use `pytorchpipeline info` to verify your paths
- Check that pipelines are in `~/.config/pytorchimagepipeline/projects/`
- For linked packages, verify the symlink exists in the projects directory

### "Missing config"

- Create or update configs in the configs directory (use `pytorchpipeline info` to see path)
- Default location: `~/.config/pytorchimagepipeline/configs/<pipeline_name>/pipeline_config.toml`
- Check the config file path with `pytorchpipeline inspect <pipeline>`

### "Invalid TOML"

- Validate your TOML syntax
- Ensure proper quoting of strings
- Check for matching braces in inline tables

### "Module not found"

- Ensure package has `__init__.py`
- Check that permanences/processes are properly imported in `__init__.py`
- Verify the symlink points to the correct directory

### Git clone fails

- Check your internet connection
- Verify repository URL
- Ensure you have access rights (for private repos)
- Try SSH if HTTPS fails (or vice versa)

---

## See Also

- [Progressive Enhancement Guide](progressive_enhancement.md) - From scripts to pipelines
- [Architecture Diagram](architecture_diagram.md) - System architecture
- [Getting Started](getting_started.md) - First steps with the pipeline
