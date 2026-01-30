# Ask AI (Flow Launcher Plugin)

Ask AI is a Flow Launcher plugin that supports multiple AI providers (Ollama, OpenAI, DeepSeek) with optional web search capabilities via Tavily. It features thinking mode to show the model's reasoning process and intelligent tool routing for web searches.

## Features

- **Multi-provider support**: Ollama (local), OpenAI, DeepSeek
- **Thinking mode**: Display model's reasoning process with `[thinking]` and `[answer]` markers
- **Web search**: Optional Tavily integration for up-to-date information
- **Smart tool routing**: Automatically triggers search when needed or use prefixes (`net`, `web`, `search`)
- **Lightweight**: Uses only `requests` library, no heavy dependencies

## Requirements

- Windows/Linux with Flow Launcher
- Python 3.10+ recommended
- **For Ollama**: Ollama running locally (Windows, WSL, or remote)
- **For OpenAI/DeepSeek**: Valid API key
- Tavily API key (optional, required for web search)

## Configuration

In Flow Launcher settings for this plugin:

### Provider Settings

- **Provider**: Select `ollama`, `openai`, or `deepseek`
- **Model**:
    - Ollama: `qwen3:4b`, `llama3`, etc.
    - OpenAI: `gpt-4o`, `o1`, `o3`, etc.
    - DeepSeek: `deepseek-reasoner`, `deepseek-chat`, etc.
- **API Base URL**:
    - Ollama: `http://localhost:11434` (default) or WSL IP like `http://172.24.32.1:11434`
    - OpenAI: `https://api.openai.com` (default)
    - DeepSeek: `https://api.deepseek.com` (default)
- **API Key**: Required for OpenAI and DeepSeek, leave empty for Ollama

### Feature Settings

- **Enable Thinking**: Show model's reasoning process (supports Ollama, DeepSeek, OpenAI o1/o3)
- **Enable Web Search**: Allow automatic web search via Tavily
- **Tavily API Key**: Your Tavily API key (required for web search)
- **Max Results**: Number of search results (1-3)
- **Use System Proxy**: Enable if behind a corporate proxy

## Usage

### Basic Questions

```
ai what is function calling
ai 什么是函数调用
```

### Web Search (Manual Trigger)

Use prefixes to force web search:

```

### Provider Issues
- **Ollama error**: Make sure Ollama is running and the host is reachable
- **OpenAI/DeepSeek error**: Verify API key is correct and has sufficient credits
- **Model not found**:
  - Ollama: Run `ollama pull <model_name>`
  - OpenAI/DeepSeek: Check model name spelling

### Web Search Issues
- **Search not working**: Ensure Tavily API key is set in settings
- **Search timeout**: Default timeout is 20 seconds, check network connection
- **Results not showing**: Make sure "Enable Web Search" is turned on

### Other Issues
- **Settings not showing**: Clear Flow Launcher cache and restart
- **Proxy issues**: Enable "Use System Proxy" if behind corporate proxy
- **Thinking not showing**: Enable "Enable Thinking" in settings

## Files
- `main.py`: Core plugin logic with multi-provider support
- `plugin.json`: Plugin metadata
- `SettingsTemplate.yaml`: Settings UI configuration
- `requirements.txt`: Python dependencies (flowlauncher, requests)reasoning:
```

[thinking]
Let me analyze this step by step...

[answer]
Based on my reasoning, the answer is...

```

### Search Results Display
When web search is used:
```

[answer]
Here's what I found...

[Web Search Results]

- Article Title
  https://example.com
  Summary of the article...

```

## How it works
1. The plugin sends your prompt to the selected provider (Ollama/OpenAI/DeepSeek)
2. If thinking is enabled, the model shows its reasoning process
3. The model decides whether to call the `search_web` tool (or you force it with prefixes)
4. If search is triggered, the plugin queries Tavily and attaches top results
5. The model returns a final answer with search results appended

## Troubleshooting
- **No output / Ollama error**: make sure Ollama is running and the host is reachable.
- **Search not used**: ensure Tavily API key is set.
- **Settings not showing**: clear Flow Launcher cache and restart.

## Files
- `main.py`: plugin logic
- `plugin.json`: plugin metadata and settings
- `requirements.txt`: dependencies
```
