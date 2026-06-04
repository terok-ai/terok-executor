# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Re-exports the executor-only timezone helper plus the shared YAML loader.

Standalone — no terok-executor domain imports, safe to use from any layer.
Cross-package helpers (``ensure_dir``, ``podman_userns_args``, the round-trip
YAML facade, ...) live in the shared [`terok_util`][terok_util] package at the
bottom of the dependency chain.  [`yaml_load`][terok_util.yaml.yaml_load] is
re-exported here so the executor's existing
``from terok_executor._util import yaml_load`` call sites stay put now that the
loader has de-vendored to terok-util.
"""

from terok_util import yaml_load

from ._timezone import detect_host_timezone

__all__ = [
    "detect_host_timezone",
    "yaml_load",
]
