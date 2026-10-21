import os

# Importing anything under zep_mcp runs zep_mcp/__init__.py, which exits
# without a key for its backend. Constructing the Zep client makes no network
# call, so any value will do; the backend stays the default (zep).
os.environ.setdefault("ZEP_API_KEY", "test-not-used")
os.environ.pop("ZEP_MCP_BACKEND", None)
