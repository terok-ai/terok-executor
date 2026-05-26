# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor sidecar JSON contract.

The terok-sandbox OCI hook reads
``$XDG_STATE_HOME/terok/sidecar/<container-name>.json`` on container
start and spawns one supervisor process per container.  This module
verifies that
[`write_supervisor_sidecar`][terok_executor.container.sidecar.write_supervisor_sidecar]
emits exactly the keys the supervisor's loader expects.
"""

from __future__ import annotations

import json
from pathlib import Path

from terok_sandbox import (
    PerContainerResources,
    SandboxConfig,
    allocate_per_container_resources,
)

from terok_executor.container.sidecar import write_supervisor_sidecar


def _cfg(tmp_path: Path) -> SandboxConfig:
    """A SandboxConfig rooted at *tmp_path* so the sidecar lands in a controllable spot."""
    return SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")


def _socket_per_container(cfg: SandboxConfig, name: str) -> PerContainerResources:
    """Allocate per-container resources for *name* (socket mode)."""
    return allocate_per_container_resources(cfg, name)


class TestWriteSupervisorSidecar:
    """Sidecar JSON contract for the per-container supervisor."""

    def test_writes_socket_mode_payload(self, tmp_path: Path) -> None:
        """Socket-mode sidecar — vault/ssh paths are derived by the supervisor."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-001")
        target = write_supervisor_sidecar("agent-task-001", cfg=cfg, per_container=per_container)

        assert target is not None
        assert target == cfg.state_dir / "sidecar" / "agent-task-001.json"
        payload = json.loads(target.read_text())
        assert payload["container_name"] == "agent-task-001"
        assert payload["ipc_mode"] in ("socket", "tcp")
        assert payload["db_path"] == str(cfg.db_path)
        assert payload["scope_id"] == ""
        assert payload["project_id"] == ""
        assert payload["task_id"] == ""

    def test_propagates_scope_and_task_ids(self, tmp_path: Path) -> None:
        """Caller-provided scope / project / task IDs land in the payload."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-002")
        target = write_supervisor_sidecar(
            "agent-task-002",
            cfg=cfg,
            per_container=per_container,
            scope_id="myproj",
            project_id="proj-abc",
            task_id="task-42",
        )

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["scope_id"] == "myproj"
        assert payload["project_id"] == "proj-abc"
        assert payload["task_id"] == "task-42"

    def test_dossier_path_optional(self, tmp_path: Path) -> None:
        """``dossier_path`` is only emitted when the caller supplies one."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-003")

        target = write_supervisor_sidecar("agent-task-003", cfg=cfg, per_container=per_container)
        assert target is not None
        assert "dossier_path" not in json.loads(target.read_text())

        dossier = tmp_path / "dossier.toml"
        dossier.write_text("")
        target = write_supervisor_sidecar(
            "agent-task-004",
            cfg=cfg,
            per_container=_socket_per_container(cfg, "agent-task-004"),
            dossier_path=dossier,
        )
        assert target is not None
        assert json.loads(target.read_text())["dossier_path"] == str(dossier)

    def test_socket_mode_omits_socket_path(self, tmp_path: Path) -> None:
        """Socket mode: the supervisor derives sockets — they're NOT in the sidecar."""
        cfg = SandboxConfig(
            state_dir=tmp_path,
            vault_dir=tmp_path / "credentials",
            services_mode="socket",
        )
        per_container = _socket_per_container(cfg, "agent-task-socket")
        target = write_supervisor_sidecar("agent-task-socket", cfg=cfg, per_container=per_container)

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["ipc_mode"] == "socket"
        assert "socket_path" not in payload
        assert "ssh_signer_socket" not in payload
        assert "tcp_port" not in payload

    def test_tcp_mode_emits_per_container_ports(self, tmp_path: Path) -> None:
        """TCP mode: ``tcp_port`` + ``ssh_signer_port`` from the per-container
        allocation (NOT from cfg's singleton)."""
        cfg = SandboxConfig(
            state_dir=tmp_path,
            vault_dir=tmp_path / "credentials",
            services_mode="tcp",
        )
        per_container = allocate_per_container_resources(cfg, "agent-task-tcp")
        target = write_supervisor_sidecar("agent-task-tcp", cfg=cfg, per_container=per_container)

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["ipc_mode"] == "tcp"
        assert payload["tcp_port"] == per_container.token_broker_port
        assert payload["ssh_signer_port"] == per_container.ssh_signer_port
        assert isinstance(payload["tcp_port"], int)
        assert "socket_path" not in payload
