# --- Setup for testing ---
# Create a temporary directory that exists for 'base_dir' validation
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from tipi.template_manager import ProjectSetup


@pytest.fixture
def base_dir():
    temp_dir = Path(tempfile.gettempdir()) / "pydantic_test_project_base"
    temp_dir.mkdir(exist_ok=True)  # Ensure it exists
    return temp_dir


def test_valid_data(base_dir):
    data = {
        "name": "MyProject",
        "base_dir": base_dir,
        "example": "basic",
        "license_type": "GPLv3",
    }

    setup = ProjectSetup(**data)
    assert setup.name == "MyProject"
    assert setup.base_dir.exists()
    assert setup.example == "basic"
    assert setup.license_type == "GPLv3"


def test_invalid_base_dir():
    data = {
        "name": "MyProject",
        "example": "basic",
        # Pass a path that definitely doesn't exist
        "base_dir": Path("/not_existing/non_existent_path_123456789"),
    }

    with pytest.raises(ValidationError) as exc:
        ProjectSetup(**data)
    assert exc.type == ValidationError
    assert exc.value.error_count() == 1
    assert exc.value.errors()[0]["type"] == "value_error"
    assert exc.value.errors()[0]["loc"] == ("base_dir",)


def test_invalid_example(base_dir):
    data = {
        "name": "MyProject",
        "base_dir": base_dir,
        "example": "unknown_example",  # Invalid name
    }

    with pytest.raises(ValidationError) as exc:
        ProjectSetup(**data)
    assert exc.type == ValidationError
    assert exc.value.error_count() == 1
    assert exc.value.errors()[0]["type"] == "literal_error"
    assert exc.value.errors()[0]["loc"] == ("example",)


def test_invalid_lincense(base_dir):
    data = {
        "name": "MyProject",
        "base_dir": base_dir,
        "example": "basic",
        "license_type": "Proprietary",  # Invalid license
    }

    with pytest.raises(ValidationError) as exc:
        ProjectSetup(**data)
    assert exc.type == ValidationError
    assert exc.value.error_count() == 1
    assert exc.value.errors()[0]["type"] == "literal_error"
    assert exc.value.errors()[0]["loc"] == ("license_type",)
