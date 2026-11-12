"""
Configuration loading utilities.

Design goals (per project spec, Section 30):
  - No hardcoded research parameters.
  - Support ${ENV_VAR} substitution (primarily EICU_DATA_ROOT).
  - Support layered configs: base.yaml + {debug,subset,full}.yaml override.
  - Fail loudly if a required value is missing, rather than silently
    defaulting to something that could change scientific methodology.

Kept deliberately dependency-light (stdlib + PyYAML only) per Section 46
("do not overengineer the first version").
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised for invalid or incomplete configuration."""


def _resolve_env_vars(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} tokens found in strings.

    Unresolved variables are left as an explicit sentinel string
    (``__UNRESOLVED__:VAR_NAME``) rather than silently becoming "None" or
    an empty path, so downstream validation can catch misconfiguration
    (e.g. EICU_DATA_ROOT not exported) instead of failing confusingly deep
    inside a pandas read_csv call.
    """
    if isinstance(value, str):
        def _sub(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                return f"__UNRESOLVED__:{var_name}"
            return env_val

        return _ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` into ``base`` (override wins on scalars)."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at top level.")
    return content


@dataclass
class RunConfig:
    """Typed, validated view over the merged configuration dict.

    This intentionally stays a thin wrapper (not a full schema library like
    Pydantic/OmegaConf) so Phase 1 has no heavy dependency footprint. As
    later phases add real parameters (model dims, federated rounds, etc.)
    this can be extended without breaking the interface: callers use
    ``cfg.raw`` for anything not yet promoted to a typed field.
    """

    raw: dict = field(default_factory=dict)
    config_path: str = ""

    @property
    def data_root(self) -> str:
        val = self.raw.get("data", {}).get("root")
        if val is None:
            raise ConfigError("Missing required config key: data.root")
        if isinstance(val, str) and val.startswith("__UNRESOLVED__:"):
            var_name = val.split(":", 1)[1]
            raise ConfigError(
                f"data.root references environment variable '{var_name}', "
                f"which is not set. Export it first, e.g.:\n"
                f"  export {var_name}=/path/to/eicu"
            )
        return val

    @property
    def mode(self) -> str:
        return self.raw.get("run", {}).get("mode", "debug")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Dotted-path getter, e.g. cfg.get('federated.rounds', 20)."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Like get(), but raises ConfigError if the key is absent.

        Use this for parameters where silently falling back to a default
        could change scientific methodology (e.g. cohort definitions,
        label thresholds) -- per project Section 30, such values must be
        explicit in the YAML, not implicit in code.
        """
        sentinel = object()
        val = self.get(dotted_key, sentinel)
        if val is sentinel:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        return val


def load_config(config_path: str, base_config_path: str | None = None) -> RunConfig:
    """Load a config, optionally layering it on top of a base config.

    Parameters
    ----------
    config_path:
        Path to the mode-specific YAML (e.g. configs/debug.yaml).
    base_config_path:
        Path to configs/base.yaml. If None, inferred as
        ``<dir of config_path>/base.yaml`` when it exists and
        config_path != base_config_path.
    """
    config_path_p = Path(config_path)
    merged: dict = {}

    if base_config_path is None:
        candidate = config_path_p.parent / "base.yaml"
        if candidate.exists() and candidate.resolve() != config_path_p.resolve():
            base_config_path = str(candidate)

    if base_config_path is not None:
        merged = _load_yaml(Path(base_config_path))

    override = _load_yaml(config_path_p)
    merged = _deep_merge(merged, override)
    merged = _resolve_env_vars(merged)

    return RunConfig(raw=merged, config_path=str(config_path_p))
