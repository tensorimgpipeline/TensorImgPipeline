#! /usr/bin/env bash

export UV_PROJECT_ENVIRONMENT=".venv-container"

echo 'export UV_PROJECT_ENVIRONMENT=".venv-container"' >> /home/vscode/.bashrc
echo 'export VIRTUAL_ENV=".venv-container"' >> /home/vscode/.bashrc

# Provide Signing Key if Available
# Make Sure the the signing key was added last
GPG_FORMAT=$(git config --get gpg.format)
if [ "$GPG_FORMAT" == "ssh" ]; then
    mkdir -p ~/.ssh && ssh-add -L | head -1 > ~/.ssh/signing_key.pub
    git config user.signingkey ~/.ssh/signing_key.pub
fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install --install-hooks
