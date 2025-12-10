# CLI Reference

Complete reference for the `tipi` command-line tool.

## Overview

The `tipi` CLI supports the complete workflow from creating new pipeline projects to running them in production.

```bash
tipi [COMMAND] [OPTIONS]
```

## Commands

### `run` - Execute a Pipeline

Run a pipeline by name.

```bash
tipi run PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to run (e.g., 'sam2segnet')

**Options:**

- `--config`, `-c` - Path to custom config file (relative to configs/)

**Examples:**

```bash
# Run default pipeline
tipi run sam2segnet

# Run with custom config
tipi run my_pipeline --config custom.toml
```

---

### `list` - List Available Pipelines

Show all available pipelines including linked subpackages.

```bash
tipi list [OPTIONS]
```

**Options:**

- `--verbose`, `-v` - Show detailed information (permanences and processes count)
- `--links/--no-links` - Show/hide symlink source paths (default: show)

**Examples:**

```bash
# Simple list
tipi list

# Detailed view
tipi list -v

# Hide link sources
tipi list --no-links
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
tipi inspect PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to inspect

**Options:**

- `--docs`, `-d` - Show docstrings for components

**Examples:**

```bash
# Basic inspection
tipi inspect sam2segnet

# With docstrings
tipi inspect sam2segnet --docs
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
tipi create PROJECT_NAME [OPTIONS]
```

**Arguments:**

- `PROJECT_NAME` - Name for the new pipeline project

**Options:**

- `--location`, `-l` - Directory to create project in (default: current directory)
- `--example`, `-e` - Include example permanence and process implementations

**Examples:**

```bash
# Create in current directory
tipi create my_pipeline

# Create in specific location
tipi create my_pipeline --location ./projects

# Create with examples
tipi create my_pipeline --example
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
tipi add SOURCE [OPTIONS]
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
tipi add ./my_pipeline_project
tipi add /path/to/project --name custom_name

# Clone from git (HTTPS)
tipi add https://github.com/user/ml-pipeline.git

# Clone from git (SSH)
tipi add git@github.com:user/pipeline.git

# Clone specific branch
tipi add https://github.com/user/repo.git --branch dev

# Clone to custom location
tipi add https://github.com/user/pipeline.git --location ./external
```

**Behavior:**

- **Local path**: Creates a symlink in the projects directory (development: `tipi/pipelines/`, production: `~/.config/tipi/projects/`)
- **Git URL**: Clones to specified location (or cache directory), then creates symlink
- Validates package structure (must have `__init__.py`)
- Checks for naming conflicts
- Confirms before overwriting existing links

---

### `remove` - Remove Linked Package

Remove a linked subpackage from the pipeline system.

```bash
tipi remove PIPELINE_NAME [OPTIONS]
```

**Arguments:**

- `PIPELINE_NAME` - Name of the linked pipeline to remove

**Options:**

- `--delete-source` - Also delete the source directory (only safe for cloned repos in submodules/)

**Examples:**

```bash
# Just remove the link
tipi remove my_pipeline

# Remove link and delete source (if in submodules/)
tipi remove my_pipeline --delete-source
```

**Safety:**

- Built-in pipelines cannot be removed
- Source deletion only works for packages in cache directory
- Confirms before deleting source directories

---

### `validate` - Validate Pipeline Configuration

Check that a pipeline is properly configured and ready to run.

```bash
tipi validate PIPELINE_NAME
```

**Arguments:**

- `PIPELINE_NAME` - Name of the pipeline to validate

**Examples:**

```bash
tipi validate sam2segnet
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

Display information about the current TensorImgPipeline installation and configuration.

```bash
tipi info
```

**Examples:**

```bash
tipi info
```

**Output:**

- Projects directory path (where pipelines are loaded from)
- Configs directory path (where configuration files are stored)
- Cache directory path (for cloned repositories)
- User config directory (~/.config/tipi)
- Directory existence status

**Default Paths:**

- Projects: `~/.config/tipi/projects/`
- Configs: `~/.config/tipi/configs/`
- Cache: `~/.cache/tipi/`

**Use Cases:**

- Debugging path issues
- Understanding where to place pipeline projects
- Checking directory structure before adding pipelines

---

## Complete Workflow Example

### Starting from Scratch

```bash
# 1. Create a new pipeline project
tipi create my_ml_pipeline --example


# 2. Navigate to the project
cd my_ml_pipeline

# 3. Edit components (in your editor)
# - Edit my_ml_pipeline/permanences.py
# - Edit my_ml_pipeline/processes.py
# - Update configs/pipeline_config.toml

# 4. Go back to main pipeline directory
cd /path/to/TensorImgPipeline

# 5. Link your project
tipi add ../my_ml_pipeline

# 6. Validate configuration
tipi validate my_ml_pipeline

# 7. Run the pipeline
tipi run my_ml_pipeline
```

### Using an Existing Git Repository

```bash
# 1. Clone and link a pipeline from GitHub
tipi add https://github.com/user/awesome-pipeline.git

# 2. Check what was added
tipi list -v

# 3. Inspect the components
tipi inspect awesome-pipeline --docs

# 4. Validate before running
tipi validate awesome-pipeline

# 5. Run it
tipi run awesome-pipeline
```

### Managing Multiple Pipelines

```bash
# List all pipelines
tipi list -v

# Inspect each one
tipi inspect pipeline1
tipi inspect pipeline2

# Run specific ones
tipi run pipeline1
tipi run pipeline2 --config custom.toml

# Remove one that's no longer needed
tipi remove old_pipeline
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

- `TIPI_CONFIG_DIR` - Override configs directory
- `TIPI_PROJECTS_DIR` - Override projects directory
- `TIPI_CACHE_DIR` - Override cache directory

**Note:** Environment variables are primarily for testing and creating isolated test scenarios.

---

## Tips & Tricks

### Quick Inspection

```bash
# Combine list and inspect for quick overview
tipi list -v && tipi inspect sam2segnet
```

### Development Workflow

```bash
# During development, frequently validate
tipi validate my_pipeline && tipi run my_pipeline
```

### Working with Git Repos

```bash
# Clone, inspect, then run
tipi add https://github.com/user/pipeline.git && \
  tipi inspect pipeline --docs && \
  tipi validate pipeline
```

### Batch Operations

```bash
# Validate all pipelines
for pipeline in $(tipi list --no-links | tail -n +3 | awk '{print $1}'); do
    echo "Validating $pipeline..."
    tipi validate $pipeline
done
```

---

## Troubleshooting

### "Pipeline not found"

- Check `tipi list` to see available pipelines
- Use `tipi info` to verify your paths
- Check that pipelines are in `~/.config/tipi/projects/`
- For linked packages, verify the symlink exists in the projects directory

### "Missing config"

- Create or update configs in the configs directory (use `tipi info` to see path)
- Default location: `~/.config/tipi/configs/<pipeline_name>/pipeline_config.toml`
- Check the config file path with `tipi inspect <pipeline>`

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
- [Getting Started](../getting_started.md) - First steps with the pipeline
