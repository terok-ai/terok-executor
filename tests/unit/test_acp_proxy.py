# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed ACP proxy.

The proxy implements both sides of the ACP protocol on one object — it
acts as an [`Agent`][acp.Agent] toward the connected client and as a
[`Client`][acp.Client] toward the in-container backend wrapper.  These
unit tests drive the typed methods directly (no JSON-RPC framing) and
patch [`spawn_agent_process`][acp.spawn_agent_process] when bind
behaviour is exercised, so no real subprocess ever starts.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from acp import RequestError
from acp.schema import (
    ConfigOptionUpdate,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
)

from terok_executor.acp import proxy as proxy_module
from terok_executor.acp.model_options import (
    build_aggregated_session_new,
    build_model_option,
    humanise_model_id,
    namespace_model_options_in_place,
)
from terok_executor.acp.proxy import (
    CLIENT_SESSION_ID,
    ACPProxy,
    AgentBindError,
)


class _StubRoster:
    """Minimal stand-in for :class:`ACPRoster`.

    The proxy only reads ``list_available_agents`` and ``wrapper_argv``;
    a thin stub keeps tests fast and isolated from sandbox plumbing.
    """

    def __init__(self, available: list[str]) -> None:
        self._available = available

    async def list_available_agents(self) -> list[str]:
        """Return the canned ``agent:model`` list."""
        return list(self._available)

    def wrapper_argv(self, agent_id: str) -> list[str]:
        """Return a sentinel argv — never exec'd in unit tests."""
        return ["echo", f"terok-{agent_id}-acp"]


class _FakeBackend:
    """Stand-in for a [`ClientSideConnection`][acp.ClientSideConnection].

    Records typed calls and returns whatever the test pre-loaded as
    canned responses.  Methods raise [`AssertionError`][] if called
    without a recorded response — the test then knows it forgot to
    mock that path.
    """

    def __init__(self, *, session_id: str = "be-1") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session_id = session_id

    async def initialize(self, **kw: Any) -> InitializeResponse:
        """Record + return a minimal-shape response."""
        self.calls.append(("initialize", kw))
        return InitializeResponse(protocol_version=kw["protocol_version"])

    async def new_session(self, **kw: Any) -> NewSessionResponse:
        """Record + return a synthetic session id."""
        self.calls.append(("new_session", kw))
        return NewSessionResponse(session_id=self.session_id)

    async def set_session_model(self, **kw: Any) -> SetSessionModelResponse:
        """Record + return an empty-body response."""
        self.calls.append(("set_session_model", kw))
        return SetSessionModelResponse()

    async def prompt(self, **kw: Any) -> PromptResponse:
        """Record + return a canned stop response."""
        self.calls.append(("prompt", kw))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, **kw: Any) -> None:
        """Record only — cancel is a notification (no reply)."""
        self.calls.append(("cancel", kw))

    async def set_config_option(self, **kw: Any) -> SetSessionConfigOptionResponse:
        """Record + return a response that echoes a bare-id model option.

        Lets the proxy's post-bind ``namespace_model_options_in_place``
        rewrite be observed end-to-end.
        """
        self.calls.append(("set_config_option", kw))
        return SetSessionConfigOptionResponse(
            config_options=[
                SessionConfigOptionSelect(
                    id="model",
                    name="Model",
                    type="select",
                    category="model",
                    current_value="opus-4.6",
                    options=[SessionConfigSelectOption(value="opus-4.6", name="Opus")],
                )
            ]
        )


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend) -> None:
    """Install *backend* as the next ``spawn_agent_process`` result."""

    @asynccontextmanager
    async def _fake_spawn(_client, _command, *_args, **_kw):
        yield backend, None

    monkeypatch.setattr(proxy_module, "spawn_agent_process", _fake_spawn)


def _new_proxy(available: list[str]) -> ACPProxy:
    """Build a proxy backed by ``_StubRoster(available)``."""
    return ACPProxy(roster=_StubRoster(available))  # type: ignore[arg-type]


class TestInitialize:
    """Pre-bind ``initialize`` is answered locally."""

    def test_returns_implementation_metadata(self) -> None:
        """Response carries protocol version and proxy identification."""
        proxy = _new_proxy([])
        resp = asyncio.run(proxy.initialize(protocol_version=1))
        assert resp.agent_info is not None
        assert resp.agent_info.name == "terok-acp"

    def test_captures_client_capabilities_for_bind_replay(self) -> None:
        """Caps from the client land on the proxy for verbatim replay on bind."""
        from acp.schema import ClientCapabilities

        proxy = _new_proxy([])
        caps = ClientCapabilities()
        asyncio.run(proxy.initialize(protocol_version=1, client_capabilities=caps))
        assert proxy._client_capabilities is caps


class TestSessionNew:
    """Pre-bind ``session/new`` aggregates the model list locally."""

    def test_returns_synthetic_session_id(self) -> None:
        """Synthetic ``proxy-1`` is returned — no backend exists yet."""
        proxy = _new_proxy(["claude:opus-4.6"])
        resp = asyncio.run(proxy.new_session(cwd="/host/proj", mcp_servers=[]))
        assert resp.session_id == CLIENT_SESSION_ID

    def test_aggregates_namespaced_model_options(self) -> None:
        """Every available ``agent:model`` appears in both ``models`` and ``configOptions``."""
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])
        resp = asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert resp.models is not None
        assert [m.model_id for m in resp.models.available_models] == [
            "claude:opus-4.6",
            "codex:gpt-5.5",
        ]
        assert resp.models.current_model_id == "claude:opus-4.6"
        assert resp.config_options is not None
        model_opt = next(opt for opt in resp.config_options if opt.category == "model")
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert [e.value for e in model_opt.options] == [
            "claude:opus-4.6",
            "codex:gpt-5.5",
        ]

    def test_rejects_second_session_new(self) -> None:
        """v1 supports one session per connection — second call errors."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert exc.value.code == -32600

    def test_remembers_default_for_lazy_bind(self) -> None:
        """The first listed model is the lazy-bind default for prompts."""
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert proxy._default_namespaced == "claude:opus-4.6"


class TestSetModelPreBind:
    """``session/set_model`` parsing before any bind."""

    def test_unnamespaced_model_id_raises_invalid_params(self) -> None:
        """Non ``agent:model`` values are rejected with -32602."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(
                proxy.set_session_model(model_id="no-namespace", session_id=CLIENT_SESSION_ID)
            )
        assert exc.value.code == -32602


class TestSetConfigOptionPreBind:
    """Older Zed sends model selection via ``set_config_option``."""

    def test_model_with_bad_namespace_raises(self) -> None:
        """Malformed ``agent:model`` short-circuits before any spawn."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(
                proxy.set_config_option(
                    config_id="model",
                    session_id=CLIENT_SESSION_ID,
                    value="no-namespace",
                )
            )
        assert exc.value.code == -32602

    def test_non_model_category_pre_bind_errors(self) -> None:
        """Pre-bind ``set_config_option`` for a non-model knob has no backend."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(
                proxy.set_config_option(
                    config_id="behavior",
                    session_id=CLIENT_SESSION_ID,
                    value="strict",
                )
            )
        assert exc.value.code == -32600


class TestPromptLazyBindGate:
    """``prompt`` lazy-binds when a default exists, otherwise errors."""

    def test_prompt_without_any_available_agent_raises(self) -> None:
        """No probed agents → no default → prompt has nothing to bind to."""
        proxy = _new_proxy([])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(proxy.prompt(prompt=[], session_id=CLIENT_SESSION_ID))
        assert exc.value.code == -32600


class TestBind:
    """End-to-end bind flow with a patched ``spawn_agent_process``."""

    def test_set_session_model_drives_backend_handshake(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First ``set_session_model`` spawns backend + replays the three frames.

        The patched backend captures the exact arguments so we can pin
        the handshake shape: namespace stripped on the way down, client
        ``cwd`` overridden to the container workspace, ``mcp_servers``
        defaulted to the empty list when the client didn't supply any.
        """
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/host/proj", mcp_servers=None)
            resp = await proxy.set_session_model(
                model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID
            )
            assert isinstance(resp, SetSessionModelResponse)

        asyncio.run(_drive())

        method_order = [name for name, _ in backend.calls]
        assert method_order == ["initialize", "new_session", "set_session_model"]
        new_session_call = backend.calls[1][1]
        assert new_session_call["cwd"] == proxy_module.CONTAINER_WORKSPACE
        assert new_session_call["mcp_servers"] == []
        set_model_call = backend.calls[2][1]
        assert set_model_call["model_id"] == "opus-4.6"  # namespace stripped
        assert set_model_call["session_id"] == backend.session_id

    def test_cross_agent_pick_after_bind_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """v1 forbids cross-agent switches; the second pick errors out."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await proxy.set_session_model(model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID)
            with pytest.raises(RequestError) as exc:
                await proxy.set_session_model(
                    model_id="codex:gpt-5.5", session_id=CLIENT_SESSION_ID
                )
            assert exc.value.code == -32602

        asyncio.run(_drive())

    def test_same_agent_repick_forwards_to_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-picking a model on the same agent forwards through stripped."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6", "claude:haiku-4.5"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await proxy.set_session_model(model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID)
            await proxy.set_session_model(model_id="claude:haiku-4.5", session_id=CLIENT_SESSION_ID)

        asyncio.run(_drive())

        set_model_calls = [kw for name, kw in backend.calls if name == "set_session_model"]
        # First call is part of bind handshake, second is the re-pick.
        assert [c["model_id"] for c in set_model_calls] == ["opus-4.6", "haiku-4.5"]

    def test_bind_failure_propagates_as_agent_bind_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backend initialize failure tears down and bubbles ``AgentBindError``."""

        class _FailingBackend(_FakeBackend):
            async def initialize(self, **_kw: Any) -> InitializeResponse:
                raise RuntimeError("simulated wrapper crash")

        _patch_spawn(monkeypatch, _FailingBackend())
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            with pytest.raises(AgentBindError):
                await proxy.set_session_model(
                    model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID
                )
            # State reset — a retry should be possible.
            assert proxy._backend is None
            assert proxy._bound_agent is None

        asyncio.run(_drive())


class TestBackendForwarding:
    """Post-bind responses get model ids re-namespaced on the way out."""

    def test_set_config_option_response_namespaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backend echoes bare ``opus-4.6``; client sees ``claude:opus-4.6``."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> SetSessionConfigOptionResponse | None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await proxy.set_session_model(model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID)
            return await proxy.set_config_option(
                config_id="theme",
                session_id=CLIENT_SESSION_ID,
                value="dark",
            )

        resp = asyncio.run(_drive())
        assert resp is not None
        model_opt = next(o for o in resp.config_options if o.category == "model")
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert model_opt.current_value == "claude:opus-4.6"
        assert [e.value for e in model_opt.options] == ["claude:opus-4.6"]


class TestNamespaceModelOptionsInPlace:
    """Typed in-place rewriter — used on every backend → client config option."""

    def test_namespaces_select_current_and_values(self) -> None:
        """Bare ``opus-4.6`` becomes ``claude:opus-4.6`` for both fields."""
        opt = SessionConfigOptionSelect(
            id="model",
            name="Model",
            type="select",
            category="model",
            current_value="opus-4.6",
            options=[
                SessionConfigSelectOption(value="opus-4.6", name="Opus"),
                SessionConfigSelectOption(value="haiku-4.5", name="Haiku"),
            ],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:opus-4.6"
        assert [e.value for e in opt.options] == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_already_namespaced_left_untouched(self) -> None:
        """Idempotent — round-tripping a proxy-built option doesn't double-prefix."""
        opt = build_model_option(["claude:opus-4.6", "claude:haiku-4.5"], current="claude:opus-4.6")
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:opus-4.6"
        assert [e.value for e in opt.options] == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_non_model_category_untouched(self) -> None:
        """Other categories pass through unchanged."""
        opt = SessionConfigOptionSelect(
            id="mode",
            name="Mode",
            type="select",
            category="mode",
            current_value="ask",
            options=[SessionConfigSelectOption(value="ask", name="Ask")],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "ask"

    def test_empty_or_none_input_is_noop(self) -> None:
        """Both ``None`` and ``[]`` inputs are accepted without error."""
        namespace_model_options_in_place(None, "claude")
        namespace_model_options_in_place([], "claude")


class TestSessionUpdateForwarding:
    """Backend → proxy → client session updates rewrite session id and model ids."""

    def test_config_option_update_namespaces_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``ConfigOptionUpdate`` carrying the model option gets namespaced.

        The proxy's ``Client.session_update`` is wired to the connected
        client side; replace the latter with a recorder to capture the
        rewritten typed update.
        """
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        captured: list[Any] = []

        class _RecordingClient:
            async def session_update(self, *, session_id: str, update: Any) -> None:
                captured.append((session_id, update))

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await proxy.set_session_model(model_id="claude:opus-4.6", session_id=CLIENT_SESSION_ID)
            # Inject the recorder *after* bind so the proxy's normal
            # initialise path isn't disturbed.
            proxy._client = _RecordingClient()  # type: ignore[assignment]
            await proxy.session_update(
                session_id=backend.session_id,
                update=ConfigOptionUpdate(
                    session_update="config_option_update",
                    config_options=[
                        SessionConfigOptionSelect(
                            id="model",
                            name="Model",
                            type="select",
                            category="model",
                            current_value="opus-4.6",
                            options=[SessionConfigSelectOption(value="opus-4.6", name="Opus")],
                        )
                    ],
                ),
            )

        asyncio.run(_drive())

        assert len(captured) == 1
        session_id, update = captured[0]
        assert session_id == CLIENT_SESSION_ID
        model_opt = update.config_options[0]
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert model_opt.current_value == "claude:opus-4.6"
        assert model_opt.options[0].value == "claude:opus-4.6"


class TestBuildHelpers:
    """The pre-bind aggregate builders."""

    def test_build_aggregated_session_new_empty_models(self) -> None:
        """Empty list yields a schema-valid response with no models block."""
        resp = build_aggregated_session_new("sess-x", [])
        assert resp.session_id == "sess-x"
        assert resp.models is None
        assert resp.config_options is None

    def test_humanise_model_id_round_trip(self) -> None:
        """The label format is ``Agent: model``."""
        assert humanise_model_id("claude:opus-4.6") == "Claude: opus-4.6"

    def test_humanise_model_id_preserves_slashes_in_model(self) -> None:
        """OpenRouter-style slash-bearing model ids survive humanisation."""
        assert humanise_model_id("opencode:opencode/big-pickle") == "Opencode: opencode/big-pickle"

    def test_humanise_unnamespaced_passes_through(self) -> None:
        """Unrecognised ids are returned verbatim — no crash."""
        assert humanise_model_id("plain") == "plain"
