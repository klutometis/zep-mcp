import os

# Importing zep_mcp.diverse runs zep_mcp/__init__.py, which exits without a
# key. Constructing the client makes no network call, so any value will do.
os.environ.setdefault("ZEP_API_KEY", "test-not-used")
