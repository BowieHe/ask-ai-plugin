import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

PLUGIN_DIR = os.path.dirname(__file__)
PLUGIN_NAME = os.path.basename(PLUGIN_DIR)
PLUGIN_ID = "7c0f4fd6-64b7-4d4e-8d7b-7f6d4b54b0a3"

LIB_DIR = os.path.join(PLUGIN_DIR, "lib")
if os.path.isdir(LIB_DIR):
    sys.path.insert(0, LIB_DIR)

from flowlauncher import FlowLauncher

try:
    import requests
except Exception as exc:
    requests = None
    REQUESTS_IMPORT_ERROR = str(exc)
else:
    REQUESTS_IMPORT_ERROR = ""


CONFIG_FILE = os.path.join(PLUGIN_DIR, "config.json")


DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "ollama",
    "model": "qwen3:4b",
    "api_base_url": "http://localhost:11434",
    "api_key": "",
    "tavily_api_key": "",
    "tavily_timeout": 20,
    "max_results": 3,
    "prompt_stop": ";;",
    "enable_web_search": False,
    "enable_thinking": True,
    "use_system_proxy": False,
    "response_preview_length": 160,
}


def get_config_value(settings: Dict[str, Any], key: str, default: Any) -> Any:
    """Get config value from settings with fallback to default"""
    value = settings.get(key)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    return value


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


def extract_thinking(response: Dict[str, Any], provider: str = "ollama") -> str:
    """Extract thinking/reasoning content from response across different providers"""
    if not isinstance(response, dict):
        return ""
    
    if provider == "ollama":
        # Ollama native format: response.message.thinking
        message = response.get("message", {})
        if isinstance(message, dict):
            thinking = message.get("thinking", "")
            if thinking:
                return thinking
        # Fallback
        return response.get("thinking", "")
    
    else:
        # OpenAI compatible format: response.choices[0].message.reasoning
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            # Check for reasoning field (OpenAI o1/o3, DeepSeek)
            reasoning = message.get("reasoning", "")
            if reasoning:
                return reasoning
            # Some providers use reasoning_content
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                return reasoning_content
    
    return ""


class ChatProvider:
    """Unified chat provider supporting multiple APIs"""
    
    # Provider configurations
    PROVIDERS = {
        "ollama": {
            "default_base_url": "http://localhost:11434",
            "endpoint": "/api/chat",
            "requires_api_key": False,
        },
        "openai": {
            "default_base_url": "https://api.openai.com",
            "endpoint": "/v1/chat/completions",
            "requires_api_key": True,
        },
        "deepseek": {
            "default_base_url": "https://api.deepseek.com",
            "endpoint": "/v1/chat/completions",
            "requires_api_key": True,
        },
    }
    
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str = "",
        api_key: str = "",
        timeout: int = 60,
        use_proxy: bool = False,
    ):
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout
        self.use_proxy = use_proxy
        
        if self.provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.PROVIDERS.keys())}")
        
        provider_config = self.PROVIDERS[self.provider]
        
        # Set base URL
        self.base_url = (base_url or provider_config["default_base_url"]).rstrip("/")
        self.endpoint = provider_config["endpoint"]
        self.api_key = api_key
        
        # Validate API key if required
        if provider_config["requires_api_key"] and not self.api_key:
            raise ValueError(f"{provider} requires an API key")
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        thinking: bool = False,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send chat request to provider"""
        if requests is None:
            raise RuntimeError(f"requests not available: {REQUESTS_IMPORT_ERROR}")
        
        if self.provider == "ollama":
            return self._chat_ollama(messages, tools, thinking, stream)
        else:
            return self._chat_openai_compatible(messages, tools, thinking, stream)
    
    def _chat_ollama(self, messages, tools, thinking, stream) -> Dict[str, Any]:
        """Ollama native API: /api/chat"""
        url = f"{self.base_url}{self.endpoint}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = tools
        
        if thinking:
            # Ollama native think parameter
            payload["think"] = True
        
        headers = {"Content-Type": "application/json"}
        
        try:
            # Create session to handle proxy settings
            session = requests.Session()
            session.trust_env = self.use_proxy
            
            response = session.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {str(e)}")
    
    def _chat_openai_compatible(self, messages, tools, thinking, stream) -> Dict[str, Any]:
        """OpenAI-compatible API: /v1/chat/completions"""
        url = f"{self.base_url}{self.endpoint}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = tools
        
        # For reasoning models, use reasoning_effort (DeepSeek, OpenAI o1)
        if thinking:
            if self.provider in ["deepseek", "openai"]:
                # DeepSeek and OpenAI o1/o3 support reasoning_effort
                payload["reasoning_effort"] = "high"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            # Create session to handle proxy settings
            session = requests.Session()
            session.trust_env = self.use_proxy
            
            response = session.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            openai_response = response.json()
            # Normalize to Ollama-like format for consistency
            return self._normalize_openai_response(openai_response)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{self.provider.title()} request failed: {str(e)}")
    
    def _normalize_openai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format to Ollama-like format for consistency"""
        choices = response.get("choices", [])
        if not choices:
            return {"message": {"role": "assistant", "content": ""}}
        
        choice = choices[0]
        message = choice.get("message", {})
        
        normalized = {
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            }
        }
        
        # Handle tool calls
        if "tool_calls" in message and message["tool_calls"]:
            normalized["message"]["tool_calls"] = message["tool_calls"]
        
        # Handle reasoning/thinking
        if "reasoning" in message:
            normalized["message"]["thinking"] = message["reasoning"]
        elif "reasoning_content" in message:
            normalized["message"]["thinking"] = message["reasoning_content"]
        
        return normalized


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_settings_fallback() -> Dict[str, Any]:
    appdata = os.getenv("APPDATA")
    if not appdata:
        return {}
    candidates = [
        os.path.join(
            appdata, "FlowLauncher", "Settings", "Plugins", PLUGIN_NAME, "settings.json"
        ),
        os.path.join(
            appdata, "FlowLauncher", "Settings", "Plugins", "Ask AI", "settings.json"
        ),
        os.path.join(
            appdata, "FlowLauncher", "Settings", "Plugins", PLUGIN_ID, "settings.json"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


class OllamaWebAgent(FlowLauncher):
    def _load_runtime_config(self) -> Dict[str, Any]:
        """Load configuration from settings with defaults"""
        settings = getattr(self, "settings", {}) or load_settings_fallback()
        
        cfg = {
            "provider": get_config_value(settings, "provider", DEFAULT_CONFIG["provider"]),
            "model": get_config_value(settings, "model", DEFAULT_CONFIG["model"]),
            "api_base_url": get_config_value(settings, "api_base_url", DEFAULT_CONFIG["api_base_url"]),
            "api_key": get_config_value(settings, "api_key", DEFAULT_CONFIG["api_key"]),
            "tavily_api_key": get_config_value(settings, "tavily_api_key", DEFAULT_CONFIG["tavily_api_key"]),
            "tavily_timeout": DEFAULT_CONFIG["tavily_timeout"],
            "max_results": get_config_value(settings, "max_results", DEFAULT_CONFIG["max_results"]),
            "prompt_stop": get_config_value(settings, "prompt_stop", DEFAULT_CONFIG["prompt_stop"]),
            "enable_web_search": get_config_value(settings, "enable_web_search", DEFAULT_CONFIG["enable_web_search"]),
            "enable_thinking": get_config_value(settings, "enable_thinking", DEFAULT_CONFIG["enable_thinking"]),
            "use_system_proxy": get_config_value(settings, "use_system_proxy", DEFAULT_CONFIG["use_system_proxy"]),
            "response_preview_length": get_config_value(settings, "response_preview_length", DEFAULT_CONFIG["response_preview_length"]),
        }
        
        # Backward compatibility
        # Use old show_thinking/thinking_mode if enable_thinking not set
        if "enable_thinking" not in settings or settings.get("enable_thinking") is None:
            show_thinking = get_config_value(settings, "show_thinking", True)
            thinking_mode = get_config_value(settings, "thinking_mode", True)
            cfg["enable_thinking"] = show_thinking and thinking_mode
        
        # Use ollama_host if set
        ollama_host = get_config_value(settings, "ollama_host", "")
        if ollama_host and cfg["provider"] == "ollama":
            cfg["api_base_url"] = ollama_host
        
        # Validate and clamp values
        try:
            cfg["max_results"] = max(1, min(3, int(cfg["max_results"])))
        except (ValueError, TypeError):
            cfg["max_results"] = DEFAULT_CONFIG["max_results"]
        
        try:
            cfg["response_preview_length"] = max(40, min(500, int(cfg["response_preview_length"])))
        except (ValueError, TypeError):
            cfg["response_preview_length"] = DEFAULT_CONFIG["response_preview_length"]
        
        return cfg

    def _chat_with_tools(
        self, prompt: str, cfg: Dict[str, Any], use_tools: bool
    ) -> str:
        if requests is None:
            raise RuntimeError(f"requests not available: {REQUESTS_IMPORT_ERROR}")
        
        # Create chat provider
        try:
            provider = ChatProvider(
                provider=cfg.get("provider", "ollama"),
                model=cfg.get("model"),
                base_url=cfg.get("api_base_url", ""),
                api_key=cfg.get("api_key", ""),
                use_proxy=to_bool(cfg.get("use_system_proxy")),
            )
        except ValueError as e:
            raise RuntimeError(str(e))
        
        tools = build_tool_schema() if use_tools else None
        last_search_summary = ""
        
        # Build system prompt
        system_content = "You are a helpful assistant."
        if use_tools:
            system_content += " You have access to a web search tool. Use it to find current information when needed."
        
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": prompt},
        ]

        thinking_enabled = to_bool(cfg.get("enable_thinking"))
        provider_name = cfg.get("provider", "ollama")

        if not use_tools:
            response = provider.chat(messages=messages, thinking=thinking_enabled)
            content = response.get("message", {}).get("content", "")
            thinking = extract_thinking(response, provider_name)
            if thinking_enabled and thinking:
                return f"[thinking]\n{thinking}\n\n[answer]\n{content}".strip()
            return content

        # Tool calling loop (increased to 3 iterations)
        for iteration in range(3):
            response = provider.chat(
                messages=messages,
                tools=tools,
                thinking=thinking_enabled
            )
            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []
            
            if not tool_calls:
                # No tool calls - return answer with search results if available
                content = message.get("content", "")
                thinking = extract_thinking(response, provider_name)
                
                if thinking_enabled and thinking:
                    answer = f"[thinking]\n{thinking}\n\n[answer]\n{content}".strip()
                else:
                    answer = content
                
                # Always append search results if we have them
                if last_search_summary:
                    answer = f"{answer}\n\n[Web Search Results]\n{last_search_summary}"
                return answer

            # Process tool calls
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
                        last_search_summary = tool_content
                    except Exception as e:
                        tool_content = f"Web search failed: {str(e)}"

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_content,
                    }
                )

        # If we exit the loop without returning
        return "Could not complete the request. Please try again."

    def query(self, param: str = "") -> List[Dict[str, Any]]:
        query = (param or "").strip()
        if not query:
            return [format_result("Type a question", "Example: ai 什么是函数调用")]

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
        if requests is None:
            return [
                format_result(
                    "Missing dependency: requests",
                    "Install in Flow Python: python -m pip install -r requirements.txt",
                )
            ]

        start = time.time()
        try:
            answer = self._chat_with_tools(query, cfg, use_tools=use_tools)
        except Exception as exc:
            message = str(exc).strip()
            provider = cfg.get("provider", "ollama")
            
            if "model" in message.lower() and "not" in message.lower() and "found" in message.lower():
                if provider == "ollama":
                    return [
                        format_result(
                            "Model not found",
                            f"Run: ollama pull {model}",
                        )
                    ]
                else:
                    return [
                        format_result(
                            "Model not found",
                            f"Check model name for {provider}",
                        )
                    ]
            
            if "api key" in message.lower() or "requires" in message.lower():
                return [
                    format_result(
                        f"{provider.title()} API key required",
                        f"Set API key in plugin settings",
                    )
                ]
            
            if not message:
                if provider == "ollama":
                    message = "Is Ollama running on localhost?"
                else:
                    message = f"Cannot connect to {provider}"
            
            return [format_result(f"{provider.title()} error", message[:200])]

        elapsed = time.time() - start
        preview_len = int(cfg.get("response_preview_length", 160))
        preview = (answer or "(empty)").strip()
        title = (
            preview
            if len(preview) <= preview_len
            else preview[:preview_len].rstrip() + "..."
        )
        subtitle = self._build_subtitle(elapsed, use_tools, query)
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

    def _build_subtitle(self, elapsed: float, use_tools: bool, query: str) -> str:
        label = ""
        if use_tools:
            label = "已联网搜索" if self._is_chinese(query) else "Web search used"
        parts = [f"{elapsed:.1f}s"]
        if label:
            parts.append(label)
        parts.append("Press Enter to open full response")
        return " | ".join(parts)

    def _is_chinese(self, text: str) -> bool:
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                return True
        return False


if __name__ == "__main__":
    OllamaWebAgent()
