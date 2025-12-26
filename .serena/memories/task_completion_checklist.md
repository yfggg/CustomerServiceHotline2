# Task Completion Checklist

- Run relevant scripts/REPL if logic changed (`python main.py` or `python mcp_service/db_server.py` as needed).
- If retrieval data touched, verify LanceDB records present and queries return results.
- For DB tools changes, test MySQL connectivity with `python mcp_service/db_pool.py` and sample queries.
- Check prompts/system messages still align with desired tone and length limits.
- Ensure hardcoded secrets/creds are handled appropriately (avoid leaking, consider env vars if modifying).
- Keep Chinese responses and formats intact (正文 + “依据：” structure).