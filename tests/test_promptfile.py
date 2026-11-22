from pathlib import Path

import pytest

from llm_playground.promptfile import PromptError, PromptLibrary, parse_prompt_file


def write(path: Path, name: str, version: str, text: str) -> Path:
    directory = path / name
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{version}.prompt"
    file.write_text(text)
    return file


SIMPLE = "---\ndescription: a test prompt\nvariables: a, b\n---\nFirst {{a}} then {{b}}.\n"


def test_parse_reads_frontmatter_and_body(tmp_path: Path) -> None:
    file = write(tmp_path, "greet", "v1", SIMPLE)
    prompt = parse_prompt_file(file)
    assert prompt.name == "greet" and prompt.version == 1
    assert prompt.description == "a test prompt" and prompt.variables == ("a", "b")
    assert prompt.template == "First {{a}} then {{b}}.\n"


def test_render_substitutes_every_declared_variable(tmp_path: Path) -> None:
    prompt = parse_prompt_file(write(tmp_path, "greet", "v1", SIMPLE))
    assert prompt.render(a="X", b="Y") == "First X then Y.\n"


def test_render_rejects_missing_and_extra_variables(tmp_path: Path) -> None:
    prompt = parse_prompt_file(write(tmp_path, "greet", "v1", SIMPLE))
    with pytest.raises(PromptError, match=r"missing value\(s\) for \['b'\]"):
        prompt.render(a="X")
    with pytest.raises(PromptError, match=r"undeclared variable\(s\) \['c'\]"):
        prompt.render(a="X", b="Y", c="Z")


def test_literal_single_braces_are_left_alone(tmp_path: Path) -> None:
    text = '---\ndescription: json example\nvariables: name\n---\nHi {{name}}, example: {"k": 1}\n'
    prompt = parse_prompt_file(write(tmp_path, "p", "v1", text))
    assert prompt.render(name="Sam") == 'Hi Sam, example: {"k": 1}\n'


def test_unused_and_undeclared_detection(tmp_path: Path) -> None:
    text = "---\ndescription: x\nvariables: a, unused\n---\nOnly {{a}} and {{typo}} appear.\n"
    prompt = parse_prompt_file(write(tmp_path, "p", "v1", text))
    assert prompt.unused_variables() == {"unused"}
    assert prompt.undeclared_placeholders() == {"typo"}


def test_variables_field_is_optional_and_defaults_to_empty(tmp_path: Path) -> None:
    text = "---\ndescription: no variables\n---\nJust static text.\n"
    prompt = parse_prompt_file(write(tmp_path, "p", "v1", text))
    assert prompt.variables == () and prompt.render() == "Just static text.\n"


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter at all\n",
        "---\ndescription: unterminated\nJust text with no closing marker\n",
        "---\nnot-a-key-value-line\n---\nbody\n",
    ],
)
def test_malformed_frontmatter_is_rejected(tmp_path: Path, text: str) -> None:
    file = write(tmp_path, "p", "v1", text)
    with pytest.raises(PromptError):
        parse_prompt_file(file)


def test_filename_must_be_v_number(tmp_path: Path) -> None:
    file = write(tmp_path, "p", "latest", SIMPLE)
    with pytest.raises(PromptError, match="filename must be"):
        parse_prompt_file(file)


def test_library_loads_every_version_and_picks_the_highest_numerically(tmp_path: Path) -> None:
    write(tmp_path, "greet", "v1", SIMPLE)
    write(tmp_path, "greet", "v2", SIMPLE)
    write(tmp_path, "greet", "v10", SIMPLE)  # numeric, not lexicographic: v10 must beat v2
    library = PromptLibrary.load(tmp_path)
    assert library.names() == ["greet"]
    assert library.versions("greet") == [1, 2, 10]
    assert library.get("greet").version == 10


def test_library_can_pin_an_older_version(tmp_path: Path) -> None:
    write(tmp_path, "greet", "v1", SIMPLE)
    write(tmp_path, "greet", "v2", SIMPLE)
    library = PromptLibrary.load(tmp_path)
    assert library.get("greet", version=1).version == 1


def test_library_render_is_a_shortcut_for_get_then_render(tmp_path: Path) -> None:
    write(tmp_path, "greet", "v1", SIMPLE)
    library = PromptLibrary.load(tmp_path)
    assert library.render("greet", a="X", b="Y") == "First X then Y.\n"


def test_library_rejects_an_unknown_name_or_version(tmp_path: Path) -> None:
    write(tmp_path, "greet", "v1", SIMPLE)
    library = PromptLibrary.load(tmp_path)
    with pytest.raises(PromptError, match="no prompt named"):
        library.get("ghost")
    with pytest.raises(PromptError, match="has no version"):
        library.get("greet", version=99)


def test_library_rejects_two_files_claiming_the_same_version(tmp_path: Path) -> None:
    # "v01" and "v1" are different filenames but the same version number once parsed as an int.
    write(tmp_path, "greet", "v1", SIMPLE)
    write(tmp_path, "greet", "v01", SIMPLE)
    with pytest.raises(PromptError, match="two files claiming version 1"):
        PromptLibrary.load(tmp_path)


def test_empty_or_missing_root_yields_an_empty_library(tmp_path: Path) -> None:
    assert PromptLibrary.load(tmp_path / "does-not-exist").names() == []
    assert PromptLibrary.load(tmp_path).names() == []  # tmp_path exists but has no prompt subdirectories


def test_iterating_a_library_yields_every_version_of_every_prompt(tmp_path: Path) -> None:
    write(tmp_path, "a", "v1", SIMPLE)
    write(tmp_path, "b", "v1", SIMPLE)
    write(tmp_path, "b", "v2", SIMPLE)
    library = PromptLibrary.load(tmp_path)
    seen = [(p.name, p.version) for p in library]
    assert seen == [("a", 1), ("b", 1), ("b", 2)]
