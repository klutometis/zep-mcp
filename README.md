# zep-mcp

Minimal MCP server for [Zep Cloud](https://www.getzep.com/) -- exposes a
temporal knowledge graph as two tools: `add_memory` and `search_memory`.

Zep Cloud is built on [Graphiti](https://github.com/getzep/graphiti), an
open-source temporal knowledge graph framework. Unlike flat memory stores,
Zep extracts entities and relationships from text, tracks how facts evolve
over time (valid/invalid timestamps), and supports hybrid search (semantic
+ BM25 keyword + graph traversal).

![Knowledge graph showing Hephaestus, golden amphipoloi, embodied LLMs, and related entities](assets/golden-amphipoloi.png)

## Tools

| Tool | Description |
|------|-------------|
| `add_memory(content)` | Add text to the knowledge graph. Zep extracts entities and relationships automatically. |
| `search_memory(query, limit=10)` | Hybrid search across the graph. Returns facts with temporal validity ranges. |

That's it. No list, no delete, no graph wipe -- the model gets the minimum
viable surface to store and retrieve facts.

## Setup

```bash
git clone git@github.com:klutometis/zep-mcp.git
cd zep-mcp
uv sync
```

## Usage

### Standalone (stdio)

```bash
ZEP_API_KEY=z_... ZEP_USER_ID=danenberg uv run python server.py
```

### With MCP Inspector

- Transport: **STDIO**
- Command: `uv`
- Args: `run --directory /path/to/zep-mcp python server.py`
- Env: `ZEP_API_KEY`, `ZEP_USER_ID`

### With an MCP gateway / client config

```json
{
  "zep": {
    "command": "uv",
    "args": ["run", "--directory", "/path/to/zep-mcp", "python", "server.py"],
    "env": {
      "ZEP_API_KEY": "z_...",
      "ZEP_USER_ID": "danenberg"
    }
  }
}
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ZEP_API_KEY` | Yes | Zep Cloud API key (starts with `z_`) |
| `ZEP_USER_ID` | No | User ID for graph scoping (default: `"default"`) |

The server creates the Zep user on startup if it doesn't exist (idempotent).

## Importing memories

A migration script is included for importing from OpenMemory (or any JSON
array of `{"content": "..."}` objects):

```bash
ZEP_API_KEY=z_... ZEP_USER_ID=danenberg \
  uv run python import-memories.py /path/to/export.json
```

Options:

| Flag | Description |
|------|-------------|
| `--nuke` | Delete user and recreate before importing (clean slate) |
| `--delay N` | Seconds between API calls (default: 0.5) |
| `--resume N` | Skip first N records (resume after partial import) |

## Architecture

```
Zep Cloud (Graphiti engine)          This server              Your client
┌────────────────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Temporal knowledge    │      │  server.py       │      │  Claude,     │
│  graph: entities,      │◄────►│  (FastMCP, stdio) │◄────►│  gateway,   │
│  relationships, facts  │      │  2 tools         │      │  Inspector   │
└────────────────────────┘      └──────────────────┘      └──────────────┘
         HTTPS                         stdio
```
