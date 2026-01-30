import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")
if os.path.isdir(LIB_DIR):
    sys.path.insert(0, LIB_DIR)

from flowlauncher import FlowLauncher

try:
    import ollama
except Exception as exc:
    ollama = None
    OLLAMA_IMPORT_ERROR = str(exc)
else:
    OLLAMA_IMPORT_ERROR = ""

try:
    import requests
except Exception as exc:
    requests = None
    REQUESTS_IMPORT_ERROR = str(exc)
else:
    REQUESTS_IMPORT_ERROR = ""


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "qwen3:4b",
    "tavily_api_key": "",
    "tavily_timeout": 8,
    "max_results": 3,
    "ollama_host": "http://localhost:11434",
    "prompt_stop": ";;",
    "enable_web_search": False,
    "show_thinking": False,
    "response_preview_length": 160,
    "use_system_proxy": False,
}


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def build_tool_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for up-to-date information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def tavily_search(query: str, api_key: str, max_results: int, timeout: int) -> str:
    if requests is None:
        raise RuntimeError(f"requests not available: {REQUESTS_IMPORT_ERROR}")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    trimmed = results[: max_results or 3]

    lines = []
    for item in trimmed:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        if content:
            content = content.strip().replace("\n", " ")
        line = f"- {title}\n  {url}\n  {content}".strip()
        lines.append(line)

    if not lines:
        return "No relevant results found."
    return "\n".join(lines)


def format_result(title: str, subtitle: str, icon: str = "icon.png") -> Dict[str, Any]:
    return {"Title": title, "SubTitle": subtitle, "IcoPath": icon}


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    start_tag = "<think>"
    end_tag = "</think>"
    while True:
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end < start:
            break
        text = text[:start] + text[end + len(end_tag) :]
    return text.strip()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_host(host: str) -> str:
    if not host:
        return host
    lowered = host.lower()
    if lowered.startswith("http://localhost"):
        return "http://127.0.0.1" + host[len("http://localhost") :]
    if lowered.startswith("https://localhost"):
        return "https://127.0.0.1" + host[len("https://localhost") :]
    if lowered.startswith("localhost"):
        return "127.0.0.1" + host[len("localhost") :]
    return host


class OllamaWebAgent(FlowLauncher):
    def _settings_value(self, key: str, fallback: Any) -> Any:
        settings = getattr(self, "settings", {}) or {}
        value = settings.get(key, "")
        if value is None or str(value).strip() == "":
            return fallback
        return value

    def _load_runtime_config(self) -> Dict[str, Any]:
        cfg = load_config()
        cfg["model"] = self._settings_value("model", cfg["model"])
        cfg["tavily_api_key"] = self._settings_value(
            "tavily_api_key", cfg["tavily_api_key"]
        )
        cfg["ollama_host"] = self._settings_value("ollama_host", cfg["ollama_host"])
        cfg["max_results"] = self._settings_value("max_results", cfg["max_results"])
        cfg["prompt_stop"] = self._settings_value("prompt_stop", cfg["prompt_stop"])
        cfg["enable_web_search"] = self._settings_value(
            "enable_web_search", cfg["enable_web_search"]
        )
        cfg["show_thinking"] = self._settings_value(
            "show_thinking", cfg["show_thinking"]
        )
        cfg["use_system_proxy"] = self._settings_value(
            "use_system_proxy", cfg["use_system_proxy"]
        )
        cfg["response_preview_length"] = self._settings_value(
            "response_preview_length", cfg["response_preview_length"]
        )
        try:
            cfg["max_results"] = max(1, min(3, int(cfg["max_results"])))
        except Exception:
            cfg["max_results"] = DEFAULT_CONFIG["max_results"]
        try:
            cfg["response_preview_length"] = max(
                40, min(500, int(cfg["response_preview_length"]))
            )
        except Exception:
            cfg["response_preview_length"] = DEFAULT_CONFIG["response_preview_length"]
        return cfg

    def _handle_config(self, args: List[str]) -> List[Dict[str, Any]]:
        cfg = load_config()
        if len(args) == 0 or args[0] in {"show", "list"}:
            return [
                format_result(
                    "Current config",
                    f"model={cfg['model']} | tavily_api_key={'set' if cfg['tavily_api_key'] else 'empty'} | max_results={cfg['max_results']}",
                )
            ]

        if args[0] == "model" and len(args) >= 2:
            cfg["model"] = " ".join(args[1:]).strip()
            save_config(cfg)
            return [format_result("Saved model", cfg["model"])]

        if args[0] == "tavily" and len(args) >= 2:
            cfg["tavily_api_key"] = " ".join(args[1:]).strip()
            save_config(cfg)
            return [format_result("Saved Tavily API key", "OK")]

        if args[0] == "max_results" and len(args) >= 2:
            try:
                cfg["max_results"] = max(1, min(3, int(args[1])))
                save_config(cfg)
                return [format_result("Saved max_results", str(cfg["max_results"]))]
            except ValueError:
                return [format_result("Invalid max_results", "Use 1-3")]

        return [
            format_result(
                "Config usage",
                "ai config model <name> | ai config tavily <key> | ai config max_results 1-3",
            )
        ]

    def _chat_with_tools(
        self, prompt: str, cfg: Dict[str, Any], use_tools: bool
    ) -> str:
        if ollama is None:
            raise RuntimeError(f"ollama not available: {OLLAMA_IMPORT_ERROR}")
        host = normalize_host(str(cfg.get("ollama_host", "")).strip())
        trust_env = to_bool(cfg.get("use_system_proxy"))
        client = (
            ollama.Client(host=host, trust_env=trust_env) if host else ollama.Client()
        )
        tools = build_tool_schema()
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        if not use_tools:
            response = client.chat(model=cfg["model"], messages=messages)
            content = response.get("message", {}).get("content", "")
            if to_bool(cfg.get("show_thinking")):
                return content
            return strip_thinking(content)

        for _ in range(2):
            response = client.chat(model=cfg["model"], messages=messages, tools=tools)
            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content", "")
                if to_bool(cfg.get("show_thinking")):
                    return content
                return strip_thinking(content)

            for call in tool_calls:
                fn = call.get("function", {})
                if fn.get("name") != "search_web":
                    continue
                args = fn.get("arguments", {})
                query = args.get("query", prompt)
                max_results = args.get("max_results", cfg["max_results"])

                if not cfg.get("tavily_api_key"):
                    tool_content = (
                        "Tavily API key is not configured. Answer without web search."
                    )
                else:
                    try:
                        tool_content = tavily_search(
                            query=query,
                            api_key=cfg["tavily_api_key"],
                            max_results=int(max_results),
                            timeout=int(cfg["tavily_timeout"]),
                        )
                    except Exception:
                        tool_content = "Web search failed. Answer without web search."

                messages.append(
                    {
                        "role": "tool",
                        "name": "search_web",
                        "content": tool_content,
                    }
                )

        return "I could not complete the tool call, please try again."

    def query(self, param: str = "") -> List[Dict[str, Any]]:
        query = (param or "").strip()
        if not query:
            return [format_result("Type a question", "Example: ai 什么是函数调用")]

        parts = query.split()
        if parts[0].lower() == "config":
            return self._handle_config(parts[1:])

        cfg = self._load_runtime_config()
        use_tools = to_bool(cfg.get("enable_web_search"))
        prompt_stop = str(cfg.get("prompt_stop", DEFAULT_CONFIG["prompt_stop"]))
        if prompt_stop and not query.endswith(prompt_stop):
            return [
                format_result(
                    "Waiting for input end",
                    f"End your query with {prompt_stop} to run",
                )
            ]
        if prompt_stop and query.endswith(prompt_stop):
            query = query[: -len(prompt_stop)].rstrip()

        lowered = query.lower()
        for prefix in ("net ", "web ", "search ", "联网 "):
            if lowered.startswith(prefix):
                use_tools = True
                query = query[len(prefix) :].strip()
                break
        model = cfg.get("model") or DEFAULT_CONFIG["model"]
        if not model:
            return [
                format_result(
                    "Model not configured",
                    "Set it in Flow Launcher plugin settings",
                )
            ]
        if ollama is None:
            return [
                format_result(
                    "Missing dependency: ollama",
                    "Install in Flow Python: python -m pip install -r requirements.txt",
                )
            ]

        start = time.time()
        try:
            answer = self._chat_with_tools(query, cfg, use_tools=use_tools)
        except Exception as exc:
            message = str(exc).strip()
            if "model" in message and "not" in message and "found" in message:
                return [
                    format_result(
                        "Model not found",
                        f"Check model name and run: ollama pull {model}",
                    )
                ]
            if not message:
                message = "Is Ollama running on localhost?"
            return [format_result("Ollama error", message[:200])]

        elapsed = time.time() - start
        preview_len = int(cfg.get("response_preview_length", 160))
        preview = (answer or "(empty)").strip()
        title = (
            preview
            if len(preview) <= preview_len
            else preview[:preview_len].rstrip() + "..."
        )
        subtitle = f"{elapsed:.1f}s | Press Enter to open full response"
        return [
            {
                "Title": title,
                "SubTitle": subtitle,
                "IcoPath": "icon.png",
                "JsonRPCAction": {
                    "method": "open_response",
                    "parameters": [preview],
                    "dontHideAfterAction": False,
                },
            }
        ]

    def open_response(self, text: str) -> bool:
        safe_text = text or "(empty)"
        tmp_dir = tempfile.gettempdir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(tmp_dir, f"ask-ai-response-{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(safe_text)
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                import subprocess

                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass
        return True


if __name__ == "__main__":
    OllamaWebAgent()
