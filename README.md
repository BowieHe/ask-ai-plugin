# Ask AI (Flow Launcher Plugin)

Ask AI is a Flow Launcher plugin that talks to a local Ollama model and can decide to use web search (Tavily) when needed. It supports function-calling style tool routing, so simple questions are answered directly and fresh questions can trigger a web search before answering.

## Features
- Local LLM via Ollama
- Optional web search via Tavily
- Configurable model and host
- Short, summarized search results (top 2-3)

## Requirements
- Windows with Flow Launcher
- Python 3.10+ recommended
- Ollama running locally (Windows or WSL)
- Tavily API key (optional but required for web search)

## Install (manual)
1. Copy the plugin folder into:
   `%APPDATA%\FlowLauncher\Plugins\Ask AI`
2. Restart Flow Launcher.

## Configuration
In Flow Launcher settings for this plugin:
- **Ollama Model**: e.g. `qwen3:4b`
- **Ollama Host** (optional): `http://<host>:11434`
- **Tavily API Key**: your Tavily key
- **Max Results**: 1-3

If Ollama runs inside WSL, set `Ollama Host` to the WSL IP, for example:
`http://172.24.32.1:11434`

## Usage
- Type: `ai <your question>`
- Example: `ai 现在北京天气如何`

## How it works
1. The plugin sends your prompt to Ollama.
2. The model decides whether to call the `search_web` tool.
3. If it calls the tool, the plugin queries Tavily and attaches the top results.
4. The model returns a final answer.

## Troubleshooting
- **No output / Ollama error**: make sure Ollama is running and the host is reachable.
- **Search not used**: ensure Tavily API key is set.
- **Settings not showing**: clear Flow Launcher cache and restart.

## Files
- `main.py`: plugin logic
- `plugin.json`: plugin metadata and settings
- `requirements.txt`: dependencies
