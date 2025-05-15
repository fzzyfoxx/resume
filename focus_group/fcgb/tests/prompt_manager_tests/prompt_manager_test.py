import pytest
from fcgb.prompt_manager import PromptManager

@pytest.fixture
def tmp_prompts(tmp_path):
    """Fixture to create a temporary prompts directory with test data."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create test prompt files
    prompt_files = {
        "test_prompt_v1.0.txt": "This is version 1.0 of test_prompt.",
        "test_prompt_v2.0.txt": "This is version 2.0 of test_prompt.",
        "another_prompt_v1.1.txt": "This is version 1.1 of another_prompt.",
        "another_prompt_v2.0.txt": "This is version 2.0 of another_prompt.",
    }

    for filename, content in prompt_files.items():
        file_path = prompts_dir / filename
        file_path.write_text(content)

    return prompts_dir

def test_prompt_manager_initialization(tmp_prompts):
    """Test initialization of PromptManager with temporary prompts."""
    version_config = {"test_prompt": "1.0", "another_prompt": None}
    pm = PromptManager(version_config, path=str(tmp_prompts))

    assert "test_prompt" in pm.prompts
    assert "another_prompt" in pm.prompts
    assert len(pm.prompts["test_prompt"]) == 2
    assert len(pm.prompts["another_prompt"]) == 2

def test_get_prompt(tmp_prompts):
    """Test the get_prompt method."""
    version_config = {"test_prompt": "2.0", "another_prompt": None}
    pm = PromptManager(version_config, path=str(tmp_prompts))

    # Get specific version
    content = pm.get_prompt("test_prompt", version="1.0")
    assert content == "This is version 1.0 of test_prompt."

    # Get default version (latest)
    content = pm.get_prompt("another_prompt")
    assert content == "This is version 2.0 of another_prompt."

    # Non-existing prompt
    with pytest.raises(ValueError, match="Prompt 'non_existing_prompt' not found."):
        pm.get_prompt("non_existing_prompt")

    # Non-existing version
    with pytest.raises(ValueError, match="Version 3.0 for prompt 'test_prompt' not found."):
        pm.get_prompt("test_prompt", version="3.0")

def test_get_prompts(tmp_prompts):
    """Test the get_prompts method."""
    version_config = {"test_prompt": "1.0", "another_prompt": None}
    pm = PromptManager(version_config, path=str(tmp_prompts))

    prompts = pm.get_prompts(["test_prompt", "another_prompt", "non_existing_prompt"])
    assert prompts["test_prompt"] == "This is version 1.0 of test_prompt."
    assert prompts["another_prompt"] == "This is version 2.0 of another_prompt."
    assert "non_existing_prompt" not in prompts

def test_prompt_versions(tmp_prompts, capsys):
    """Test the prompt_versions method."""
    version_config = {"test_prompt": "1.0", "another_prompt": None}
    pm = PromptManager(version_config, path=str(tmp_prompts))

    pm.prompt_versions("test_prompt")
    captured = capsys.readouterr()
    assert "Version 2.0" in captured.out
    assert "Version 1.0 (default)" in captured.out

    pm.prompt_versions("another_prompt")
    captured = capsys.readouterr()
    assert "Version 2.0 (default)" in captured.out
    assert "Version 1.1" in captured.out

    pm.prompt_versions("non_existing_prompt")
    captured = capsys.readouterr()
    assert "Prompt 'non_existing_prompt' not found." in captured.out

