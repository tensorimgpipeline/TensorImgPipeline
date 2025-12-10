# CLI Enhancement Summary

## Overview

The CLI has been significantly enhanced to support the research-to-production workflow with modular pipeline projects.

## New Commands

### 1. **`run`** (Updated)

- Renamed from `build_pipeline` to `run`
- Added `--config` option for custom config files
- Better error messages with Rich formatting

### 2. **`list`** (New)

- Lists all available pipelines
- Shows pipeline type (Built-in vs Linked)
- Displays config status
- Optional verbose mode (`-v`) shows component counts
- Optional `--links` flag shows symlink sources
- Rich table formatting

### 3. **`inspect`** (New)

- Detailed component inspection
- Shows all permanences and processes
- Tree view visualization
- Optional docstring display (`--docs`)
- Shows config file location and status

### 4. **`create`** (New)

- Creates standalone pipeline projects
- Generates complete project structure
- Includes pyproject.toml, README, .gitignore
- Optional example implementations (`--example`)
- Custom location support (`--location`)

### 5. **`add`** (New)

- Links local projects as subpackages
- Clones from git repositories
- Creates symlinks in pipelines directory
- Supports SSH and HTTPS git URLs
- Branch selection for git repos (`--branch`)
- Custom clone location (`--location`)
- Validates package structure

### 6. **`remove`** (New)

- Removes linked subpackages
- Safely unlinks symlinks
- Optional source deletion (`--delete-source`)
- Safety checks for built-in pipelines
- Only deletes cloned repos in submodules/

### 7. **`validate`** (New)

- Validates pipeline configuration
- Checks module imports
- Validates class inheritance
- Checks TOML syntax
- Verifies config sections
- Clear success/failure reporting

## Workflow Support

### Creating New Projects

```bash
# Create standalone project
tipi create my_pipeline --example

# Project structure created:
my_pipeline/
├── my_pipeline/              # Package
│   ├── __init__.py
│   ├── permanences.py
│   └── processes.py
├── configs/
│   └── pipeline_config.toml
├── pyproject.toml
├── README.md
└── .gitignore
```

### Adding Projects

```bash
# Link local project
tipi add ./my_pipeline

# Clone from git
tipi add https://github.com/user/pipeline.git

# Creates symlink:
# tipi/pipelines/my_pipeline -> source
```

### Managing Projects

```bash
# List all
tipi list -v

# Inspect specific
tipi inspect my_pipeline --docs

# Validate
tipi validate my_pipeline

# Run
tipi run my_pipeline

# Remove
tipi remove my_pipeline
```

## File Generators

### `_generate_project_init()`

Creates package `__init__.py` with:

- Component registration dictionaries
- Version information
- Proper imports

### `_generate_permanence_file()`

Creates `permanences.py` with:

- Base imports
- Example permanence class (optional)
- Template comments
- Lifecycle methods

### `_generate_processes_file()`

Creates `processes.py` with:

- Base imports
- Example process class (optional)
- Template comments
- Execute and skip methods

### `_generate_config_file()`

Creates TOML config with:

- Permanence definitions
- Process definitions
- Example configurations (optional)
- Inline comments

### `_generate_readme()`

Creates project README with:

- Structure overview
- Installation instructions
- Usage examples
- Development guide

### `_generate_pyproject()`

Creates `pyproject.toml` with:

- Package metadata
- Dependencies
- Development dependencies
- Ruff configuration
- Build system setup

### `_generate_gitignore()`

Creates `.gitignore` with:

- Python artifacts
- Virtual environments
- IDE files
- Testing artifacts
- OS files

## Rich Formatting

All commands use Rich library for beautiful output:

- **Tables**: List command with colored columns
- **Trees**: Inspect command with hierarchical view
- **Panels**: Success/error messages with borders
- **Colors**: Semantic coloring (green=success, red=error, yellow=warning)
- **Console**: Consistent formatting throughout

## Safety Features

### Link Management

- Validates package structure before linking
- Checks for naming conflicts
- Confirms before overwriting links
- Distinguishes between built-in and linked pipelines

### Source Deletion

- Only deletes sources in `submodules/` directory
- Requires explicit `--delete-source` flag
- Interactive confirmation before deletion
- Prevents accidental deletion of external projects

### Validation

- Module import checking
- Class inheritance validation
- TOML syntax verification
- Config section requirements
- Detailed error reporting

## Integration with Progressive Enhancement

The CLI now supports the Level 5 workflow:

```
Level 4: Functions          Level 5: Full Pipeline
─────────────────────      ────────────────────────
@pipeline_process     →    tipi create my_pipeline
def train():          →    Edit permanences.py & processes.py
    ...               →    Update config.toml
                      →    tipi add ./my_pipeline
Run as script         →    tipi run my_pipeline
```

## Example Workflows

### Researcher Creates New Pipeline

```bash
# Day 1: Create project
tipi create research_pipeline --example
cd research_pipeline

# Day 2-7: Develop (edit files)
# Edit research_pipeline/permanences.py
# Edit research_pipeline/processes.py
# Edit configs/pipeline_config.toml

# Week 2: Link to main pipeline
cd /path/to/TensorImgPipeline
tipi add ../research_pipeline

# Validate and run
tipi validate research_pipeline
tipi run research_pipeline
```

### Team Member Uses Existing Pipeline

```bash
# Clone from team repo
tipi add https://github.com/team/ml-pipeline.git

# Inspect what it does
tipi inspect ml-pipeline --docs

# Run it
tipi validate ml-pipeline
tipi run ml-pipeline
```

### Cleanup Old Projects

```bash
# List all pipelines
tipi list -v

# Remove old one (keep source)
tipi remove old_pipeline

# Remove cloned repo (delete source)
tipi remove temp_pipeline --delete-source
```

## Benefits

1. **Modular**: Projects are self-contained and reusable
2. **Flexible**: Link local projects or clone from git
3. **Safe**: Multiple validation and confirmation steps
4. **User-friendly**: Rich formatting and clear messages
5. **Complete**: Full project scaffolding
6. **Discoverable**: List and inspect commands
7. **Maintainable**: Easy to add/remove projects

## Future Enhancements

Possible future additions:

- `tipi update <name>` - Update git-cloned packages
- `tipi test <name>` - Run pipeline tests
- `tipi export <name>` - Export pipeline as standalone package
- `tipi template list` - Show available templates
- `tipi template create <name>` - Create custom templates
- Auto-completion support
- Configuration validation with JSON schema
- Pipeline dependency resolution
- Version management

## Documentation

Created documentation:

- `docs/cli_reference.md` - Complete CLI reference
- Updated `docs/architecture_diagram.md` - Architecture overview
- Updated `docs/progressive_enhancement.md` - Level 5 workflow

## Technical Details

### Symlink Management

- Uses `Path.symlink_to()` for cross-platform compatibility
- Resolves symlinks with `Path.resolve()`
- Checks symlink status with `Path.is_symlink()`

### Git Integration

- Uses `subprocess.run()` for git commands
- Supports HTTPS and SSH URLs
- Branch selection with `--branch` flag
- Error handling with stderr capture

### Module Loading

- Uses `importlib.import_module()` for dynamic loading
- Handles ImportError gracefully
- Validates class inheritance with `issubclass()`
- Accesses module attributes with `getattr()`

### File Generation

- Template strings with f-strings
- Conditional content based on flags
- Consistent formatting and style
- Includes comments and documentation
