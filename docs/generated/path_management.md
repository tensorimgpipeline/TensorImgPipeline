# Path Management and Deployment

This document explains how PytorchImagePipeline manages paths and how the PathManager enables flexible project organization.

## Overview

PytorchImagePipeline uses a centralized path management system that:

1. Stores all pipelines in user-configurable directories
2. Follows XDG Base Directory standards on Linux/Unix
3. Supports environment variable overrides for testing and custom setups

## PathManager

The `PathManager` class (in `pytorchimagepipeline/paths.py`) provides appropriate paths for:

- **Projects directory**: Where pipeline packages are loaded from
- **Configs directory**: Where TOML configuration files are stored
- **Cache directory**: Where cloned git repositories are stored

## Directory Structure

### Default Paths

```
# User's home directory
~/.config/pytorchimagepipeline/
├── projects/                   # ← Projects loaded from here
│   ├── my_pipeline/           # User's custom pipeline
│   └── cloned_pipeline/       # symlink to cache
├── configs/                    # ← Configs loaded from here
│   ├── my_pipeline/
│   │   └── pipeline_config.toml
│   └── cloned_pipeline/
│       └── pipeline_config.toml

~/.cache/pytorchimagepipeline/  # ← Git clones stored here
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
pytorchpipeline info
```

This shows:

- All directory paths
- Directory existence status

### Getting Started

```bash
# 1. Install the package
pip install pytorchimagepipeline

# 2. Check configuration
pytorchpipeline info

# 3. Create a new pipeline
pytorchpipeline create my_pipeline

# 4. Link it
pytorchpipeline add ./my_pipeline
# - Or use 'pytorchpipeline add' to link external projects
```

### Production Workflow (End Users)

```bash
# 1. Install from PyPI
pip install pytorchimagepipeline

# Or use uvx for isolated execution
uvx pytorchimagepipeline --help

# 2. Check mode (should show "production")
pytorchpipeline info

# 3. Create or add custom pipelines
# Option A: Create new pipeline
pytorchpipeline create my_pipeline
cd my_pipeline
# ... develop your pipeline ...
cd ..
pytorchpipeline add ./my_pipeline

# Option B: Clone from git
pytorchpipeline add https://github.com/user/awesome-pipeline.git

# 4. Run your pipeline
pytorchpipeline run my_pipeline
```

### Hybrid Workflow (Production with Local Development)

```bash
# 1. Install from PyPI for CLI tools
pip install pytorchimagepipeline

# 2. Develop pipelines separately
mkdir ~/ml-pipelines
cd ~/ml-pipelines
pytorchpipeline create image_segmentation
cd image_segmentation
# ... develop ...

# 3. Link to production installation
pytorchpipeline add ~/ml-pipelines/image_segmentation

# 4. Use like any other pipeline
pytorchpipeline list
pytorchpipeline run image_segmentation
```

## Environment Variables

For advanced use cases or testing, you can override the automatic detection:

### Override Specific Directories

```bash
# Override projects directory
export PYTORCHPIPELINE_PROJECTS_DIR=/custom/path/projects
pytorchpipeline list

# Override configs directory
export PYTORCHPIPELINE_CONFIG_DIR=/custom/path/configs
pytorchpipeline run my_pipeline

# Override cache directory
export PYTORCHPIPELINE_CACHE_DIR=/custom/path/cache
pytorchpipeline add https://github.com/user/repo.git
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

1. **Create standalone projects** that can be added via `pytorchpipeline add`

   ```bash
   pytorchpipeline create my_pipeline
   cd my_pipeline
   # Edit your pipeline code
   ```

2. **Link your project** for testing

   ```bash
   pytorchpipeline add ./my_pipeline
   pytorchpipeline validate my_pipeline
   ```

3. **Use environment overrides** for isolated testing

   ```bash
   PYTORCHPIPELINE_PROJECTS_DIR=/tmp/test_projects pytorchpipeline list
   ```

### For End Users

1. **Installation**: Use pip or uvx

   ```bash
   pip install pytorchimagepipeline
   # or
   uvx pytorchimagepipeline
   ```

2. **Custom Pipelines**: Store in `~/.config/pytorchimagepipeline/projects/`

   - Create with `pytorchpipeline create`
   - Or add existing with `pytorchpipeline add`

3. **Verification**: Always check paths with `pytorchpipeline info` when troubleshooting

### For System Administrators

1. **Multi-User Setup**: Each user gets their own `~/.config/pytorchimagepipeline/`

2. **Shared Pipelines**: Use git repositories

   ```bash
   # Each user can clone
   pytorchpipeline add https://company.com/shared-pipeline.git
   ```

3. **Custom Paths**: Set system-wide environment variables if needed
   ```bash
   # In /etc/environment or similar
   PYTORCHPIPELINE_PROJECTS_DIR=/opt/ml-pipelines/projects
   PYTORCHPIPELINE_CONFIG_DIR=/opt/ml-pipelines/configs
   ```

## Migration Guide

### Packaging Your Pipeline for Distribution

When you're ready to distribute your pipeline:

1. **Package your pipeline** as a standalone project:

   ```bash
   pytorchpipeline create my_pipeline_standalone
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
   pytorchpipeline add https://github.com/you/my_pipeline_standalone.git
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
   pytorchpipeline add ./my_package
   ```

## Troubleshooting

### Pipeline Not Found

```bash
# Check configuration
pytorchpipeline info

# List available pipelines
pytorchpipeline list

# Check ~/.config/pytorchimagepipeline/projects/
```

### Config Not Found

```bash
# Check config directory
pytorchpipeline info

# Inspect specific pipeline
pytorchpipeline inspect my_pipeline

# Create config in ~/.config/pytorchimagepipeline/configs/my_pipeline/
```

### Import Errors

```bash
# Verify module structure
pytorchpipeline inspect my_pipeline

# Check that __init__.py exists and has proper registration
ls -la ~/.config/pytorchimagepipeline/projects/my_pipeline/

# Validate the pipeline
pytorchpipeline validate my_pipeline
```

### Symlink Issues (Linux/Mac)

```bash
# Check symlink
ls -la ~/.config/pytorchimagepipeline/projects/

# Recreate if broken
pytorchpipeline remove my_pipeline
pytorchpipeline add /path/to/my_pipeline
```

### Permission Issues

```bash
# Ensure proper ownership
ls -la ~/.config/pytorchimagepipeline/

# Fix if needed
chmod -R u+rwX ~/.config/pytorchimagepipeline/
```

## API Reference

For advanced usage in Python code:

```python
from pytorchimagepipeline.paths import get_path_manager

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
- [Installation Guide](installation.md) - Installation instructions
- [Getting Started](getting_started.md) - First steps with pipelines
