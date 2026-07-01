from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import urllib.error
import urllib.request


@dataclass
class QwenRuntimeConfig:
    model_name: str = "Qwen2.5-VL-72B-Instruct"
    backend_mode: str = "openai_compatible"
    endpoint_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    api_key_env: str = "QWEN_VL_API_KEY"
    model_root: str = "models/Qwen2.5-VL-72B-Instruct"
    lora_adapter_root: str = "adapters/aero_lora"
    timeout_seconds: int = 60
    enable_remote_call: bool = False
    fallback_mode: str = "local_kgmcf_planner"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QwenRuntimeConfig":
        base = cls()
        for key in base.__dataclass_fields__:
            if key in data and data[key] is not None:
                setattr(base, key, data[key])
        return base

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend_mode": self.backend_mode,
            "endpoint_url": self.endpoint_url,
            "api_key_env": self.api_key_env,
            "model_root": self.model_root,
            "lora_adapter_root": self.lora_adapter_root,
            "timeout_seconds": self.timeout_seconds,
            "enable_remote_call": self.enable_remote_call,
            "fallback_mode": self.fallback_mode,
        }


def load_runtime_config(root: Path) -> QwenRuntimeConfig:
    path = root / "configs" / "model_runtime.json"
    if not path.exists():
        return QwenRuntimeConfig()
    with path.open("r", encoding="utf-8") as f:
        return QwenRuntimeConfig.from_dict(json.load(f))


class QwenRuntimeClient:
    """HTTP client for a locally deployed Qwen2.5-VL endpoint.

    The project does not ship large model weights. Inference is enabled when a
    local or private server is configured and `enable_remote_call` is true.
    Otherwise the caller can safely fall back to the deterministic KGMCF planner.
    """

    def __init__(self, root: Path, config: Optional[QwenRuntimeConfig] = None):
        self.root = root.resolve()
        self.config = config or load_runtime_config(root)

    def status(self) -> Dict[str, Any]:
        model_path = (self.root / self.config.model_root).resolve()
        adapter_path = (self.root / self.config.lora_adapter_root).resolve()
        return {
            **self.config.as_dict(),
            "model_path_exists": model_path.exists(),
            "adapter_path_exists": adapter_path.exists(),
            "model_path": str(model_path),
            "adapter_path": str(adapter_path),
            "api_key_configured": bool(os.environ.get(self.config.api_key_env, "")),
            "call_enabled": bool(self.config.enable_remote_call),
        }

    def test_connection(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
        url = endpoint_url or self.config.endpoint_url
        if not self.config.enable_remote_call and endpoint_url is None:
            return {
                "status": "configured_only",
                "message": "Remote inference is disabled in configs/model_runtime.json. Configure an endpoint and enable remote calls to test a deployed Qwen2.5-VL-72B service.",
                "endpoint_url": url,
            }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": "Return the word ready."}],
            "temperature": 0,
            "max_tokens": 8,
        }
        try:
            data = self._post_json(url, payload, api_key=api_key)
            return {"status": "connected", "endpoint_url": url, "response": data}
        except Exception as exc:
            return {"status": "unreachable", "endpoint_url": url, "message": str(exc)}

    def chat(self, messages: List[Dict[str, Any]], endpoint_url: Optional[str] = None, api_key: Optional[str] = None, max_tokens: int = 512) -> Dict[str, Any]:
        if not self.config.enable_remote_call and endpoint_url is None:
            return {
                "status": "fallback_required",
                "message": "Qwen2.5-VL remote call is not enabled. The prototype will use the local KGMCF planner unless a model endpoint is configured.",
            }
        url = endpoint_url or self.config.endpoint_url
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            data = self._post_json(url, payload, api_key=api_key)
            return {"status": "ok", "endpoint_url": url, "response": data}
        except Exception as exc:
            return {"status": "error", "endpoint_url": url, "message": str(exc)}

    def _post_json(self, url: str, payload: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        token = api_key or os.environ.get(self.config.api_key_env, "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=int(self.config.timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}
