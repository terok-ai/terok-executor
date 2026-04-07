# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from layered YAML config into a queryable roster.

Delegates to :mod:`.loader` for YAML deserialization and roster construction.
:mod:`.config_stack` provides generic layered config resolution used by the
loader internally and by external consumers (terok config stack composition).
"""

from .loader import (
    AgentRoster,
    CredentialProxyRoute,
    MountDef,
    SidecarSpec,
    ensure_proxy_routes,
    get_roster,
    load_roster,
)

__all__ = [
    "AgentRoster",
    "CredentialProxyRoute",
    "MountDef",
    "SidecarSpec",
    "ensure_proxy_routes",
    "get_roster",
    "load_roster",
]
