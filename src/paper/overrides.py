"""`--set a.b.c=value` command-line overrides (value parsed as YAML), e.g. --set paper.window.min_icu_los_minutes=1440."""

from __future__ import annotations

import yaml


def apply_set_overrides(raw: dict, items) -> dict:
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        node = raw; parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = node[part] = {}
            node = nxt
        node[parts[-1]] = yaml.safe_load(val)
    return raw
