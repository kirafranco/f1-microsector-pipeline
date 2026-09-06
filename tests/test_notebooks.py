"""F014: what a versioned notebook must look like, checked without a stack.

Global CLAUDE.md 3.4 says notebooks are versioned with outputs cleared unless a
specific output is the point of the notebook. That is a rule about the file on
disk, so it is checked structurally here rather than trusted to a habit: a
notebook committed with its outputs carries whatever was on screen that day --
often a row of real data, sometimes a connection string in a traceback -- and
makes every diff a wall of base64.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

NOTEBOOKS = PROJECT_ROOT / "notebooks"


def notebook_paths() -> list[Path]:
    return sorted(p for p in NOTEBOOKS.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]


class TestThereIsSomethingToCheck:
    def test_the_notebooks_directory_exists(self) -> None:
        assert NOTEBOOKS.is_dir()

    def test_the_smoke_notebook_is_versioned(self) -> None:
        """F014's acceptance vehicle; -m docker executes it against the stack."""
        assert (NOTEBOOKS / "00_stack_smoke.ipynb").exists()


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda p: p.name)
class TestEveryNotebook:
    def test_it_is_valid_json_in_the_current_format(self, path: Path) -> None:
        notebook = load(path)
        assert notebook.get("nbformat") == 4
        assert notebook.get("nbformat_minor", 0) >= 5, "nbformat 4.5+ gives cells ids"

    def test_outputs_are_cleared(self, path: Path) -> None:
        """Global 3.4. A stored output is data in the repository."""
        with_outputs = [c.get("id", i) for i, c in enumerate(code_cells(load(path))) if c.get("outputs")]
        assert with_outputs == [], f"{path.name}: cells still carry outputs: {with_outputs}"

    def test_execution_counts_are_cleared(self, path: Path) -> None:
        """Left behind, they make a diff out of the order someone ran cells in."""
        numbered = [c.get("id", i) for i, c in enumerate(code_cells(load(path)))
                    if c.get("execution_count") is not None]
        assert numbered == [], f"{path.name}: cells still numbered: {numbered}"

    def test_every_cell_has_an_id(self, path: Path) -> None:
        """nbformat 4.5 requires them; without one, nbconvert warns on every run."""
        missing = [i for i, c in enumerate(load(path).get("cells", [])) if not c.get("id")]
        assert missing == [], f"{path.name}: cells without an id at {missing}"

    def test_it_names_a_kernel(self, path: Path) -> None:
        assert load(path).get("metadata", {}).get("kernelspec", {}).get("name"), path.name

    def test_no_credential_is_written_into_it(self, path: Path) -> None:
        """Credentials reach a kernel from the environment, never from a cell."""
        body = path.read_text(encoding="utf-8")
        for marker in ("POSTGRES_PASSWORD=", "postgresql://f1_", "JUPYTER_TOKEN="):
            assert marker not in body, f"{path.name}: looks like a credential ({marker})"

    def test_it_does_not_import_from_another_notebook(self, path: Path) -> None:
        """Global 3.4: reusable logic is promoted to src/, not imported sideways."""
        source = "\n".join("".join(c.get("source", [])) for c in code_cells(load(path)))
        assert "import_ipynb" not in source and "%run " not in source, path.name
