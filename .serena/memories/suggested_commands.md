# Suggested Commands (Windows)

- Run chat REPL: `python main.py`
- Start MCP DB server (stdio; default for chain): `python mcp_service/db_server.py`
- Start MCP DB server via HTTP (optional debug): `python mcp_service/db_server.py --http`
- Quick DB self-test (prints one row if available): `python mcp_service/db_pool.py`
- Inspect LanceDB contents (Python REPL):
  ```python
  from app.vector_db import VectorDatabase; vd=VectorDatabase(); vd.fetch_all_documents()
  ```
- Populate LanceDB with sample records: run the commented snippet in `app/vector_db.py` or call `vd.add_records([...])` in Python.