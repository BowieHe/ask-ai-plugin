import json
import os
import time
from typing import Any, Dict, List

import ollama
import requests
from flowlauncher import FlowLauncher


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "qwen2.5-coder:14b",
    "tavily_api_key": "",
    "tavily_timeout": 8,
    "max_results": 3,
    "ollama_host": "",
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
        try:
            cfg["max_results"] = max(1, min(3, int(cfg["max_results"])))
        except Exception:
            cfg["max_results"] = DEFAULT_CONFIG["max_results"]
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

    def _chat_with_tools(self, prompt: str, cfg: Dict[str, Any]) -> str:
        host = str(cfg.get("ollama_host", "")).strip()
        client = ollama.Client(host=host) if host else ollama.Client()
        tools = build_tool_schema()
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. If a user question needs fresh info, call search_web.",
            },
            {"role": "user", "content": prompt},
        ]

        for _ in range(2):
            response = client.chat(model=cfg["model"], messages=messages, tools=tools)
            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message.get("content", "")

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
        model = cfg.get("model") or DEFAULT_CONFIG["model"]
        if not model:
            return [
                format_result(
                    "Model not configured",
                    "Set it in Flow Launcher plugin settings",
                )
            ]

        start = time.time()
        try:
            answer = self._chat_with_tools(query, cfg)
        except Exception:
            return [
                format_result(
                    "Ollama connection failed", "Is Ollama running on localhost?"
                )
            ]

        elapsed = time.time() - start
        subtitle = f"{elapsed:.1f}s"
        return [format_result(answer or "(empty)", subtitle)]


if __name__ == "__main__":
    OllamaWebAgent()
