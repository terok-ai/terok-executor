# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor sidecar JSON writer.

The terok-sandbox OCI hook (installed by ``terok-sandbox setup``)
spawns one supervisor process per container at start.  The hook is
triggered by — and reads from — the ``terok.sandbox.sidecar`` OCI
annotation; the annotation's value is the absolute path to the JSON
written here.

Schema mirrors the sandbox writer ([`terok_sandbox.launch._write_sidecar`][terok_sandbox.launch._write_sidecar]):
keys ``container_name``, ``ipc_mode`` (``"socket"`` or ``"tcp"``),
``db_path``, ``scope_id``, ``project_id``, ``task_id``, plus one of
``socket_path`` or ``tcp_port`` depending on the IPC mode.
The caller (``AgentRunner.launch_prepared``) emits the returned
path as the OCI annotation so the hook can find this file.

Path: ``<cfg.state_dir>/sidecar/<container-name>.json``.  The
single ``sidecar/`` segment is the canonical location — no XDG
guessing, no nested ``terok/`` infix — and matches what the
``terok-sandbox`` writer also emits.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from terok_executor.integrations.sandbox import PerContainerResources, SandboxConfig

_logger = logging.getLogger(__name__)


def write_supervisor_sidecar(
    container_name: str,
    *,
    cfg: SandboxConfig,
    per_container: PerContainerResources,
    scope_id: str = "",
    project_id: str = "",
    task_id: str = "",
    dossier_path: Path | str | None = None,
) -> Path | None:
    """Persist the per-container supervisor sidecar JSON.

    Best-effort: a write failure is logged to stderr and returns
    ``None`` — the supervisor will refuse to spawn (no sidecar = no
    terok-managed container) but the launch itself isn't blocked.
    Mirrors the soft-fail policy of
    [`terok_sandbox.launch._write_sidecar`][terok_sandbox.launch._write_sidecar].

    Args:
        container_name: The ``--name`` passed to ``podman run`` (and
            therefore the sidecar filename key).
        cfg: Sandbox config — sources ``state_dir``, ``services_mode``,
            ``db_path``, plus the active transport's socket path or
            broker port.
        scope_id: Credential scope; empty for non-scoped runs.
        project_id: Terok project ID; empty when not under a project.
        task_id: Terok task ID; empty for standalone executor runs.
        dossier_path: Optional path to the per-task dossier file the
            shield reads; ``None`` for executor runs that don't carry
            a dossier.

    Returns:
        The written sidecar path, or ``None`` if the write failed.
    """
    sidecar_dir = cfg.state_dir / "sidecar"
    try:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"warning: sidecar dir setup failed: {exc}", file=sys.stderr)
        return None

    payload: dict[str, object] = {
        "container_name": container_name,
        "ipc_mode": cfg.services_mode,
        "db_path": str(cfg.db_path),
        "scope_id": scope_id or "",
        "project_id": project_id or "",
        "task_id": task_id or "",
        # The supervisor runs in crun's rootless userns where geteuid==0;
        # ``namespace_runtime_dir`` would misroute to ``/run/terok``.
        "runtime_dir": str(cfg.runtime_dir),
    }
    if cfg.services_mode == "tcp":
        payload["tcp_port"] = per_container.token_broker_port
        payload["ssh_signer_port"] = per_container.ssh_signer_port
    if dossier_path is not None:
        payload["dossier_path"] = str(dossier_path)

    target = sidecar_dir / f"{container_name}.json"
    try:
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"warning: sidecar write failed: {exc}", file=sys.stderr)
        return None

    _logger.debug("Wrote supervisor sidecar for %s → %s", container_name, target)
    return target
