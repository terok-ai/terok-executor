# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary and typed builders for ACP's model selector.

The host proxy hides multiple in-container agents behind a single ACP
endpoint by namespacing each agent's model ids as ``agent:model``.
This module owns three things on top of the ACP SDK's pydantic models:

- the namespace separator and the human-readable label rendering,
- the pre-bind ``session/new`` aggregate builder ([`build_aggregated_session_new`][terok_executor.acp.model_options.build_aggregated_session_new]),
- the post-bind in-place rewrite that puts the ``agent:`` prefix back on
  the bare model ids a bound backend emits in
  ``configOptions[category=model]``
  ([`namespace_model_options_in_place`][terok_executor.acp.model_options.namespace_model_options_in_place]).
"""

from __future__ import annotations

from acp import NewSessionResponse
from acp.schema import (
    ModelInfo,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionModelState,
)

MODEL_OPTION_CATEGORY = "model"
"""ACP semantic category for the model selector configOption."""

MODEL_NAMESPACE_SEP = ":"
"""Separator between agent and model in the namespaced id (e.g.
``claude:opus-4.6``).  Chosen over ``/`` to avoid collisions with
OpenRouter-style ids like ``anthropic/claude-opus-4``."""


def humanise_model_id(namespaced: str) -> str:
    """Render ``claude:opus-4.6`` as ``Claude: opus-4.6`` for the picker.

    Colon matches the wire-level [`MODEL_NAMESPACE_SEP`][terok_executor.acp.model_options.MODEL_NAMESPACE_SEP]
    so an OpenCode-style ``opencode:opencode/big-pickle`` reads as one
    provider plus one slash-bearing model id.  Forwards verbatim if the
    input isn't a namespaced pair.
    """
    agent, _, model = namespaced.partition(MODEL_NAMESPACE_SEP)
    if not agent or not model:
        return namespaced
    return f"{agent.capitalize()}: {model}"


def build_model_option(namespaced_models: list[str], *, current: str) -> SessionConfigOptionSelect:
    """Build a ``category: "model"`` select option with namespaced ids."""
    return SessionConfigOptionSelect(
        id="model",
        name="Model",
        type="select",
        description="AI model to use",
        category=MODEL_OPTION_CATEGORY,
        current_value=current,
        options=[
            SessionConfigSelectOption(value=ident, name=humanise_model_id(ident))
            for ident in namespaced_models
        ],
    )


def build_aggregated_session_new(session_id: str, models: list[str]) -> NewSessionResponse:
    """Construct the pre-bind ``session/new`` reply for *models*.

    Empty *models* yields a schema-valid response with no ``models`` or
    ``configOptions`` block (both have non-nullable required fields the
    proxy can't fill in for an empty list).
    """
    if not models:
        return NewSessionResponse(session_id=session_id)
    current = models[0]
    return NewSessionResponse(
        session_id=session_id,
        models=SessionModelState(
            available_models=[
                ModelInfo(model_id=ident, name=humanise_model_id(ident)) for ident in models
            ],
            current_model_id=current,
        ),
        config_options=[build_model_option(models, current=current)],
    )


def namespace_model_options_in_place(
    config_options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean] | None,
    bound_agent: str,
) -> None:
    """Mutate any ``category: "model"`` select so values become ``agent:value``.

    Used on every backend → client frame that carries
    ``configOptions[*]`` post-bind (``ConfigOptionUpdate`` notification,
    ``SetSessionConfigOptionResponse``).  Already-namespaced values are
    left alone so the function is idempotent — paths the proxy itself
    constructed (e.g. ack of a model pick) round-trip cleanly.
    """
    if not config_options or not bound_agent:
        return
    prefix = f"{bound_agent}{MODEL_NAMESPACE_SEP}"
    for opt in config_options:
        if not isinstance(opt, SessionConfigOptionSelect):
            continue
        if opt.category != MODEL_OPTION_CATEGORY and opt.id != MODEL_OPTION_CATEGORY:
            continue
        if MODEL_NAMESPACE_SEP not in opt.current_value:
            opt.current_value = prefix + opt.current_value
        for entry in opt.options:
            if isinstance(entry, SessionConfigSelectOption):
                if MODEL_NAMESPACE_SEP not in entry.value:
                    entry.value = prefix + entry.value
            elif isinstance(entry, SessionConfigSelectGroup):
                for sub in entry.options:
                    if MODEL_NAMESPACE_SEP not in sub.value:
                        sub.value = prefix + sub.value
