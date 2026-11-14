"""The unfilled state of the files Phil fills in during the campaign (review S5, H1).

Three files of the repository are templates until a step of RUNBOOK.md fills them in:

- `configs/recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset until the
  smoke runs (§2.2);
- `configs/data.yaml` leaves the dataset and tokenizer hashes unset until the corpus is
  tokenized (§2.1);
- `configs/commitments.yaml` declares nothing until the gate of §3.4 (see
  tests/fixtures/commitments.py for that one).

A test of what the harness does while one of them is unfilled builds that state here. It does
not read it off the repository, or the tier would turn red at the commit the runbook asks for.
Nothing in this module is a result.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src import config

# (section, key) of the values plan §7.1 and §7.3 leave to the smoke runs.
UNSET_RECIPE_KEYS = (("batch", "micro_batch"), ("grad_scaler", "init_scale"), ("grad_scaler", "growth_interval"))
UNSET_RECIPE_PATHS = frozenset(f"recipe.{section}.{key}" for section, key in UNSET_RECIPE_KEYS)

_REAL_LOAD_RECIPE = config.load_recipe  # the loader itself, before any test replaces it


def unfilled_recipe(monkeypatch) -> dict:
    """Make `config.load_recipe()` return the repository's recipe with the three smoke-run values unset.

    Every other value is the repository's own. Returns one such recipe.
    """

    def load() -> dict:
        recipe = _REAL_LOAD_RECIPE()
        for section, key in UNSET_RECIPE_KEYS:
            recipe[section][key] = None
        return recipe

    monkeypatch.setattr(config, "load_recipe", load)
    return load()


def unpinned_data_config(directory: Path) -> Path:
    """A copy of `configs/data.yaml` with both hashes unset, written into `directory`."""
    data = yaml.safe_load((config.CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    data["dataset_sha256"] = None
    data["tokenizer"]["sha256"] = None
    path = Path(directory) / "unpinned_data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
