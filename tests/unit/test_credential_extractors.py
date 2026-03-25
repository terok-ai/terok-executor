# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-provider credential extractors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terok_agent.credential_extractors import (
    extract_api_key_env,
    extract_claude_oauth,
    extract_codex_oauth,
    extract_credential,
    extract_gh_token,
    extract_glab_token,
    extract_json_api_key,
)


class TestClaudeOAuth:
    """Verify Claude credential extraction."""

    def test_extracts_access_token(self, tmp_path: Path) -> None:
        """Extracts accessToken from .credentials.json."""
        cred = {
            "claudeAiOauth": {
                "token": {
                    "accessToken": "sk-ant-test-123",
                    "refreshToken": "rt-test-456",
                    "expiresAt": 1700000000,
                }
            }
        }
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_claude_oauth(tmp_path)
        assert result["access_token"] == "sk-ant-test-123"
        assert result["refresh_token"] == "rt-test-456"
        assert result["type"] == "oauth"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_claude_oauth(tmp_path)

    def test_empty_token_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when accessToken is empty."""
        (tmp_path / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"token": {}}}))
        with pytest.raises(ValueError, match="no accessToken"):
            extract_claude_oauth(tmp_path)


class TestCodexOAuth:
    """Verify Codex credential extraction."""

    def test_extracts_access_token(self, tmp_path: Path) -> None:
        """Extracts access_token from auth.json."""
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": "sk-openai-test", "refresh_token": "rt-oai"}})
        )
        result = extract_codex_oauth(tmp_path)
        assert result["access_token"] == "sk-openai-test"
        assert result["type"] == "oauth"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_codex_oauth(tmp_path)


class TestApiKeyEnv:
    """Verify dotenv-style API key extraction."""

    def test_extracts_named_var(self, tmp_path: Path) -> None:
        """Extracts a specific named variable."""
        (tmp_path / ".env").write_text("MISTRAL_API_KEY=test-key-123\n")
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "test-key-123"
        assert result["type"] == "api_key"

    def test_strips_quotes(self, tmp_path: Path) -> None:
        """Strips surrounding quotes from values."""
        (tmp_path / ".env").write_text("MISTRAL_API_KEY='quoted-key'\n")
        result = extract_api_key_env(tmp_path, ".env", "MISTRAL_API_KEY")
        assert result["key"] == "quoted-key"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_api_key_env(tmp_path, ".env", "X")


class TestJsonApiKey:
    """Verify JSON config API key extraction."""

    def test_extracts_api_key(self, tmp_path: Path) -> None:
        """Extracts api_key from config.json."""
        (tmp_path / "config.json").write_text(json.dumps({"api_key": "blab-key-123"}))
        result = extract_json_api_key(tmp_path)
        assert result["key"] == "blab-key-123"

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when api_key field is absent."""
        (tmp_path / "config.json").write_text(json.dumps({"other": "value"}))
        with pytest.raises(ValueError, match="No api_key"):
            extract_json_api_key(tmp_path)


class TestGhToken:
    """Verify GitHub CLI token extraction."""

    def test_extracts_oauth_token(self, tmp_path: Path) -> None:
        """Extracts oauth_token from hosts.yml."""
        (tmp_path / "hosts.yml").write_text(
            "github.com:\n  oauth_token: ghp_test123\n  user: testuser\n"
        )
        result = extract_gh_token(tmp_path)
        assert result["token"] == "ghp_test123"
        assert result["host"] == "github.com"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when file is missing."""
        with pytest.raises(ValueError, match="not found"):
            extract_gh_token(tmp_path)


class TestGlabToken:
    """Verify GitLab CLI token extraction."""

    def test_extracts_token(self, tmp_path: Path) -> None:
        """Extracts token from config.yml hosts section."""
        (tmp_path / "config.yml").write_text(
            "hosts:\n  gitlab.com:\n    token: glpat-test456\n    api_host: gitlab.com\n"
        )
        result = extract_glab_token(tmp_path)
        assert result["token"] == "glpat-test456"
        assert result["host"] == "gitlab.com"


class TestExtractCredential:
    """Verify the dispatch function."""

    def test_dispatches_to_claude(self, tmp_path: Path) -> None:
        """extract_credential('claude', ...) calls extract_claude_oauth."""
        cred = {"claudeAiOauth": {"token": {"accessToken": "sk-test"}}}
        (tmp_path / ".credentials.json").write_text(json.dumps(cred))
        result = extract_credential("claude", tmp_path)
        assert result["access_token"] == "sk-test"

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        """Unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="No credential extractor"):
            extract_credential("unknown-agent", tmp_path)
