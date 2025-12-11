# TensorImgPipeline

> [!CAUTION]
> The Project has moved. It is now available at:
> [TensorImgPipeline](https://pypi.org/project/TensorImgPipeline/)

![LOGO](docs/assets/logos/tipi_logo_text.v2.png)

[![Release](https://img.shields.io/github/v/release/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/v/release/tensorimgpipeline/TensorImgPipeline)
[![Build status](https://img.shields.io/github/actions/workflow/status/tensorimgpipeline/TensorImgPipeline/main.yml?branch=main)](https://github.com/tensorimgpipeline/TensorImgPipeline/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tensorimgpipeline/TensorImgPipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/tensorimgpipeline/TensorImgPipeline)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/commit-activity/m/tensorimgpipeline/TensorImgPipeline)
[![License](https://img.shields.io/github/license/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/license/tensorimgpipeline/TensorImgPipeline)

This is a repository for creating and running Tensor Image Pipelines, short tipis.

- **Github repository**: <https://github.com/tensorimgpipeline/TensorImgPipeline/>
- **Documentation** <https://tensorimgpipeline.github.io/TensorImgPipeline/>

## Development

To contribute to this project it is strongly recommended to use the devcontainer for local testing.
Since some tests involve creating demo projects, which could lead to tampering with your local environment outside of the project.

If your contribution does not involve in certain actions, you might also execute run tests on a local test environment.
For Example: Your tests includes only backend actions, which do not interact with the app config or create new scaffolding projects.

If you change files, which may interact with certain files, please execute all tests inside devcontainer and extend tests if necessary.

> To run all tests local execute the test like this (WARNING: this interacts with files outside this project):

```bash
IS_IN_CONTAINER=true uv run pytest
```

### Setup devcontainer

- Ensure docker or podman (recommended) are installed and configured correctly.
- Install VScode Extensions for Remote Development

This project already provides the necessary configurations to run this project inside devcontainer:

- Open Command Pallete.
- Execute `Dev Containers: Reopen in Container`

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:tensorimgpipeline/TensorImgPipeline.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/tensorimgpipeline/TensorImgPipeline/settings/secrets/actions/new).
- Create a [new release](https://github.com/tensorimgpipeline/TensorImgPipeline/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
