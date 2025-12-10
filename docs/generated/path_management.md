# Path Management and Deployment

This document explains how TensorImgPipeline manages paths and how the PathManager enables flexible project organization.

## Overview

TensorImgPipeline uses a centralized path management system that:

1. Stores all pipelines in user-configurable directories
2. Follows XDG Base Directory standards on Linux/Unix
3. Supports environment variable overrides for testing and custom setups

## PathManager

The `PathManager` class (in `tipi/paths.py`) provides appropriate paths for:

- **Projects directory**: Where pipeline packages are loaded from
- **Configs directory**: Where TOML configuration files are stored
- **Cache directory**: Where cloned git repositories are stored

## Directory Structure

### Default Paths

```
# User's home directory
~/.config/tipi/
├── projects/                   # ← Projects loaded from here
│   ├── my_pipeline/           # User's custom pipeline
│   └── cloned_pipeline/       # symlink to cache
├── configs/                    # ← Configs loaded from here
│   ├── my_pipeline/
│   │   └── pipeline_config.toml
│   └── cloned_pipeline/
│       └── pipeline_config.toml

~/.cache/tipi/  # ← Git clones stored here
└── projects/
    └── cloned_pipeline/
```

## XDG Base Directory Specification

In production mode, PathManager follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html):

- `XDG_CONFIG_HOME` (default: `~/.config`): For configuration and project links
- `XDG_CACHE_HOME` (default: `~/.cache`): For temporary data like git clones

This ensures proper integration with Linux/Unix systems and respects user preferences.

## Usage Examples

### Check Your Current Configuration

```bash
tipi info
```

This shows:

- All directory paths
- Directory existence status

### Getting Started

```bash
# 1. Install the package
pip install tensorimgpipeline

# 2. Check configuration
tipi info

# 3. Create a new pipeline
tipi create my_pipeline

# 4. Link it
tipi add ./my_pipeline
# - Or use 'tipi add' to link external projects
```

### Production Workflow (End Users)

```bash
# 1. Install from PyPI
pip install tensorimgpipeline

# Or use uvx for isolated execution
uvx tensorimgpipeline --help

# 2. Check mode (should show "production")
tipi info

# 3. Create or add custom pipelines
# Option A: Create new pipeline
tipi create my_pipeline
cd my_pipeline
# ... develop your pipeline ...
cd ..
tipi add ./my_pipeline

# Option B: Clone from git
tipi add https://github.com/user/awesome-pipeline.git

# 4. Run your pipeline
tipi run my_pipeline
```

### Hybrid Workflow (Production with Local Development)

```bash
# 1. Install from PyPI for CLI tools
pip install tensorimgpipeline

# 2. Develop pipelines separately
mkdir ~/ml-pipelines
cd ~/ml-pipelines
tipi create image_segmentation
cd image_segmentation
# ... develop ...

# 3. Link to production installation
tipi add ~/ml-pipelines/image_segmentation

# 4. Use like any other pipeline
tipi list
tipi run image_segmentation
```

## Environment Variables

For advanced use cases or testing, you can override the automatic detection:

### Override Specific Directories

```bash
# Override projects directory
export TIPI_PROJECTS_DIR=/custom/path/projects
tipi list

# Override configs directory
export TIPI_CONFIG_DIR=/custom/path/configs
tipi run my_pipeline

# Override cache directory
export TIPI_CACHE_DIR=/custom/path/cache
tipi add https://github.com/user/repo.git
```

**Note:** Environment variables are useful for testing and creating isolated environments.

## Module Import System

The PathManager handles dynamic module imports by:

1. Adding the projects directory to `sys.path`
2. Using standard Python import mechanisms

```python
# Dynamically adds projects directory to sys.path
# Then imports my_pipeline
module = path_manager.import_project_module("my_pipeline")
# Module is now available just like any installed package
```

This allows seamless imports regardless of where pipelines are located.

## Best Practices

### For Pipeline Developers

1. **Create standalone projects** that can be added via `tipi add`

   ```bash
   tipi create my_pipeline
   cd my_pipeline
   # Edit your pipeline code
   ```

2. **Link your project** for testing

   ```bash
   tipi add ./my_pipeline
   tipi validate my_pipeline
   ```

3. **Use environment overrides** for isolated testing

   ```bash
   TIPI_PROJECTS_DIR=/tmp/test_projects tipi list
   ```

### For End Users

1. **Installation**: Use pip or uvx

   ```bash
   pip install tensorimgpipeline
   # or
   uvx tensorimgpipeline
   ```

2. **Custom Pipelines**: Store in `~/.config/tipi/projects/`

   - Create with `tipi create`
   - Or add existing with `tipi add`

3. **Verification**: Always check paths with `tipi info` when troubleshooting

### For System Administrators

1. **Multi-User Setup**: Each user gets their own `~/.config/tipi/`

2. **Shared Pipelines**: Use git repositories

   ```bash
   # Each user can clone
   tipi add https://company.com/shared-pipeline.git
   ```

3. **Custom Paths**: Set system-wide environment variables if needed
   ```bash
   # In /etc/environment or similar
   TIPI_PROJECTS_DIR=/opt/ml-pipelines/projects
   TIPI_CONFIG_DIR=/opt/ml-pipelines/configs
   ```

## Migration Guide

### Packaging Your Pipeline for Distribution

When you're ready to distribute your pipeline:

1. **Package your pipeline** as a standalone project:

   ```bash
   tipi create my_pipeline_standalone
   # Copy your code to the new project
   ```

2. **Publish to git**:

   ```bash
   cd my_pipeline_standalone
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <url>
   git push
   ```

3. **Users can now add it**:
   ```bash
   tipi add https://github.com/you/my_pipeline_standalone.git
   ```

### From Standalone to Integrated

If you have an existing Python package:

1. **Ensure proper structure**:

   ```
   my_package/
   ├── my_package/
   │   ├── __init__.py        # Must have this
   │   ├── permanences.py
   │   └── processes.py
   └── configs/
       └── pipeline_config.toml
   ```

2. **Add registration in `__init__.py`**:

   ```python
   from my_package.permanences import MyPermanence
   from my_package.processes import MyProcess

   permanences_to_register = {
       "MyPermanence": MyPermanence,
   }

   processes_to_register = {
       "MyProcess": MyProcess,
   }
   ```

3. **Link it**:
   ```bash
   tipi add ./my_package
   ```

## Troubleshooting

### Pipeline Not Found

```bash
# Check configuration
tipi info

# List available pipelines
tipi list

# Check ~/.config/tipi/projects/
```

### Config Not Found

```bash
# Check config directory
tipi info

# Inspect specific pipeline
tipi inspect my_pipeline

# Create config in ~/.config/tipi/configs/my_pipeline/
```

### Import Errors

```bash
# Verify module structure
tipi inspect my_pipeline

# Check that __init__.py exists and has proper registration
ls -la ~/.config/tipi/projects/my_pipeline/

# Validate the pipeline
tipi validate my_pipeline
```

### Symlink Issues (Linux/Mac)

```bash
# Check symlink
ls -la ~/.config/tipi/projects/

# Recreate if broken
tipi remove my_pipeline
tipi add /path/to/my_pipeline
```

### Permission Issues

```bash
# Ensure proper ownership
ls -la ~/.config/tipi/

# Fix if needed
chmod -R u+rwX ~/.config/tipi/
```

## API Reference

For advanced usage in Python code:

```python
from tipi.paths import get_path_manager

# Get the global PathManager instance
pm = get_path_manager()

# Get directories
projects_dir = pm.get_projects_dir()
configs_dir = pm.get_configs_dir()
cache_dir = pm.get_cache_dir()

# Get config path for specific pipeline
config_path = pm.get_config_path("my_pipeline", "pipeline_config.toml")

# Import a project module
module = pm.import_project_module("my_pipeline")

# Get configuration info
info = pm.get_info()
for key, value in info.items():
    print(f"{key}: {value}")

# Setup Python path (automatically called by import_project_module)
pm.setup_python_path("my_pipeline")
```

## See Also

- [CLI Reference](cli_reference.md) - Complete CLI command documentation
- [Installation Guide](../installation.md) - Installation instructions
- [Getting Started](../getting_started.md) - First steps with pipelines
