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
        # OpenAI compatible format: response.choices[0].message.reasoning or reasoning_content
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            # Check for reasoning field (OpenAI o1/o3, DeepSeek Chat)
            reasoning = message.get("reasoning", "")
            if reasoning:
                return reasoning
            # Some providers use reasoning_content (DeepSeek Reasoner)
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                return reasoning_content
        
        # Also check if message is at top level (some formats)
        message = response.get("message", {})
        if isinstance(message, dict):
            reasoning = message.get("reasoning", "")
            if reasoning:
                return reasoning
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                return reasoning_content
    
    return ""


class ChatProvider:
    """Unified chat provider supporting multiple APIs"""
    
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: int = 60,
        use_proxy: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.use_proxy = use_proxy
        
        # Auto-detect provider based on URL
        self.is_ollama = self._is_ollama_url(base_url)
        
        if self.is_ollama:
            self.endpoint = "/api/chat"
            self.provider_name = "ollama"
        else:
            self.endpoint = "/v1/chat/completions"
            self.provider_name = "openai-compatible"
            if not self.api_key:
                raise ValueError("API key is required for cloud providers")
    
    @staticmethod
    def _is_ollama_url(url: str) -> bool:
        """Check if URL is a local Ollama instance"""
        url_lower = url.lower()
        return "localhost" in url_lower or "127.0.0.1" in url_lower or "0.0.0.0" in url_lower
    
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
        
        if self.is_ollama:
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
            # OpenAI-compatible providers may support reasoning_effort
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
            # Get more detailed error info
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    if isinstance(error_detail, dict):
                        error_msg = f"{error_msg} | Details: {error_detail.get('error', error_detail)}"
                except Exception:
                    pass
            raise RuntimeError(f"API request failed: {error_msg}")
    
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
        
        # Handle reasoning/thinking - preserve original field names for API compatibility
        if "reasoning" in message:
            normalized["message"]["thinking"] = message["reasoning"]
            normalized["message"]["reasoning"] = message["reasoning"]
        elif "reasoning_content" in message:
            normalized["message"]["thinking"] = message["reasoning_content"]
            normalized["message"]["reasoning_content"] = message["reasoning_content"]
        
        return normalized


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_settings_fallback() -> Dict[str, Any]:
    """Load settings from Flow Launcher settings file"""
    candidates = []
    
    # Windows paths
    appdata = os.getenv("APPDATA")
    if appdata:
        candidates.extend([
            os.path.join(
                appdata, "FlowLauncher", "Settings", "Plugins", PLUGIN_NAME, "settings.json"
            ),
            os.path.join(
                appdata, "FlowLauncher", "Settings", "Plugins", "Ask AI", "settings.json"
            ),
            os.path.join(
                appdata, "FlowLauncher", "Settings", "Plugins", PLUGIN_ID, "settings.json"
            ),
        ])
    
    # Linux paths
    home = os.path.expanduser("~")
    if home:
        candidates.extend([
            os.path.join(home, ".config", "FlowLauncher", "Settings", "Plugins", PLUGIN_NAME, "settings.json"),
            os.path.join(home, ".config", "FlowLauncher", "Settings", "Plugins", "Ask AI", "settings.json"),
            os.path.join(home, ".config", "FlowLauncher", "Settings", "Plugins", PLUGIN_ID, "settings.json"),
        ])
    
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
        
        # Debug: Write settings to temp file to see what we got
        try:
            debug_path = os.path.join(tempfile.gettempdir(), "ask-ai-debug-settings.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({
                    "has_self_settings": hasattr(self, "settings"),
                    "self_settings": getattr(self, "settings", None),
                    "loaded_settings": settings,
                    "APPDATA": os.getenv("APPDATA"),
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        
        cfg = {
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
                model=cfg.get("model"),
                base_url=cfg.get("api_base_url"),
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
            system_content += " You have access to a web search tool. Use it to find current information when needed. You can search up to 3 times. After gathering sufficient information, provide your final answer without making more searches."
        
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": prompt},
        ]

        thinking_enabled = to_bool(cfg.get("enable_thinking"))
        provider_name = "ollama" if provider.is_ollama else "openai"

        if not use_tools:
            response = provider.chat(messages=messages, thinking=thinking_enabled)
            content = response.get("message", {}).get("content", "")
            thinking = extract_thinking(response, provider_name)
            if thinking_enabled and thinking:
                return f"[thinking]\n{thinking}\n\n[answer]\n{content}".strip()
            return content

        # Tool calling loop (increased to 10 iterations for complex reasoning models like DeepSeek Reasoner)
        # Clear old debug file if exists
        debug_path = os.path.join(tempfile.gettempdir(), "ask-ai-debug-messages.json")
        try:
            if os.path.exists(debug_path):
                os.remove(debug_path)
        except Exception:
            pass
        
        debug_log = []  # Collect debug info
        
        for iteration in range(10):
            response = provider.chat(
                messages=messages,
                tools=tools,
                thinking=thinking_enabled
            )
            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []
            
            # Debug: Log the full response and message for this iteration
            debug_log.append({
                "iteration": iteration + 1,
                "full_response": response,
                "full_message": message,
                "num_tool_calls": len(tool_calls),
                "provider_name": provider_name,
            })
            
            if not tool_calls:
                # No tool calls - return answer with search results if available
                content = message.get("content", "")
                thinking = extract_thinking(response, provider_name)
                
                # Write debug log before returning
                try:
                    with open(debug_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "total_iterations": len(debug_log),
                            "completed_normally": True,
                            "extracted_thinking": thinking,
                            "thinking_enabled": thinking_enabled,
                            "provider": provider_name,
                            "iterations": debug_log,
                        }, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                
                if thinking_enabled and thinking:
                    answer = f"[thinking]\n{thinking}\n\n[answer]\n{content}".strip()
                else:
                    answer = content
                
                # Always append search results if we have them
                if last_search_summary:
                    answer = f"{answer}\n\n[Web Search Results]\n{last_search_summary}"
                return answer

            # Model wants to use tools - add the assistant message with tool_calls first
            # This is required for OpenAI-compatible APIs
            assistant_content = message.get("content")
            
            # For reasoning models (like deepseek-reasoner), content might be null/empty
            # when calling tools. DeepSeek API requires a non-null content field.
            if assistant_content is None or (isinstance(assistant_content, str) and not assistant_content.strip()):
                # Provide a meaningful placeholder when using tools
                assistant_content = "I will use the search tool to find this information."
            
            assistant_msg = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls
            }
            
            # For DeepSeek Reasoner, preserve reasoning_content field if present
            if "reasoning_content" in message:
                assistant_msg["reasoning_content"] = message["reasoning_content"]
            elif "reasoning" in message:
                assistant_msg["reasoning"] = message["reasoning"]
            
            messages.append(assistant_msg)

            # Process tool calls
            for call in tool_calls:
                fn = call.get("function", {})
                if fn.get("name") != "search_web":
                    continue
                
                # Get tool_call_id (required for OpenAI-compatible APIs)
                tool_call_id = call.get("id", "")
                
                # Handle arguments - might be string or dict
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                
                query = args.get("query", prompt) if isinstance(args, dict) else prompt
                max_results = args.get("max_results", cfg["max_results"]) if isinstance(args, dict) else cfg["max_results"]

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

                # Build tool response message
                tool_message = {
                    "role": "tool",
                    "content": tool_content,
                }
                
                # Add tool_call_id for OpenAI-compatible APIs
                if not provider.is_ollama and tool_call_id:
                    tool_message["tool_call_id"] = tool_call_id
                
                messages.append(tool_message)

        # Write debug log after max iterations
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_iterations": len(debug_log),
                    "max_iterations_reached": True,
                    "provider": provider_name,
                    "thinking_enabled": thinking_enabled,
                    "iterations": debug_log,
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # If we exit the loop without returning, check if we have search results
        # If yes, return with results; if not, try one final attempt without tools
        if last_search_summary:
            # We have search results, return them directly
            return f"Based on the search results, here is the information I found:\n\n[Web Search Results]\n{last_search_summary}"
        
        # No search results yet - make one final attempt without tools
        try:
            # Add a clear instruction to stop using tools and provide final answer
            messages.append({
                "role": "user",
                "content": "Based on the search results above, please provide your final answer now. Do not make any more tool calls."
            })
            
            final_response = provider.chat(
                messages=messages,
                tools=None,  # No tools for final answer
                thinking=thinking_enabled
            )
            final_message = final_response.get("message", {})
            content = final_message.get("content", "")
            thinking = extract_thinking(final_response, provider_name)
            
            # Clean up any DSML markers that might leak through
            if content:
                # Remove DeepSeek's internal tool calling format if it appears
                import re
                content = re.sub(r'<｜DSML｜[^>]*>.*?</｜DSML｜[^>]*>', '', content, flags=re.DOTALL)
                content = content.strip()
            
            if thinking_enabled and thinking:
                answer = f"[thinking]\n{thinking}\n\n[answer]\n{content}".strip()
            else:
                answer = content
            
            if last_search_summary:
                answer = f"{answer}\n\n[Web Search Results]\n{last_search_summary}"
            
            return answer if answer else "Could not complete the request. Please try again."
        except Exception:
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
            base_url = cfg.get("api_base_url", "")
            is_ollama = "localhost" in base_url.lower() or "127.0.0.1" in base_url.lower()
            
            if "model" in message.lower() and "not" in message.lower() and "found" in message.lower():
                if is_ollama:
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
                            f"Check model name in your provider",
                        )
                    ]
            
            if "api key" in message.lower() or "requires" in message.lower():
                return [
                    format_result(
                        "API key required",
                        "Set API key in plugin settings",
                    )
                ]
            
            if not message:
                if is_ollama:
                    message = "Is Ollama running on localhost?"
                else:
                    message = "Cannot connect to API"
            
            return [format_result("API error", message)]

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
