# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-agent."""

from __future__ import annotations

import argparse
import sys


def _cmd_agents(args: argparse.Namespace) -> None:
    """List registered agents (and optionally tools)."""
    from .registry import get_registry

    reg = get_registry()
    names = reg.all_names if args.all else reg.agent_names

    if not names:
        print("No agents registered.", file=sys.stderr)
        return

    # Collect rows: (name, label, kind/tier)
    rows: list[tuple[str, str, str]] = []
    for name in sorted(names):
        p = reg.providers.get(name)
        auth = reg.auth_providers.get(name)
        label = p.label if p else (auth.label if auth else name)
        if p:
            rows.append((name, label, f"tier {_tier(name)}"))
        else:
            rows.append((name, label, "tool"))

    # Column widths
    w_name = max(len(r[0]) for r in rows)
    w_label = max(len(r[1]) for r in rows)

    header = f"{'NAME':<{w_name}}  {'LABEL':<{w_label}}  TYPE"
    print(header)
    for name, label, kind in rows:
        print(f"{name:<{w_name}}  {label:<{w_label}}  {kind}")


def _tier(name: str) -> int:
    """Read tier from the raw YAML data for an agent."""
    from .registry import _load_bundled_agents, _load_user_agents

    raw = _load_bundled_agents()
    raw.update(_load_user_agents())
    return raw.get(name, {}).get("tier", 0)


def main() -> None:
    """Run the terok-agent CLI."""
    parser = argparse.ArgumentParser(prog="terok-agent", description="Single-agent task runner")
    sub = parser.add_subparsers(dest="command")

    # agents
    agents_p = sub.add_parser("agents", help="List registered agents")
    agents_p.add_argument("--all", action="store_true", help="Include tools (gh, glab)")

    args = parser.parse_args()
    if args.command == "agents":
        _cmd_agents(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
