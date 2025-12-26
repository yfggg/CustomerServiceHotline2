# CustomerServiceHotline2 Overview

- Purpose: Chinese AI客服助手 combining retrieval (LanceDB + BM25) and MCP tools to query MySQL for orders/company info; CLI loop in `main.py`.
- Entry point: `python main.py` launches REPL that routes user input to `ChatAssistant.stream_chat`.
- Core flow: build LangChain pipeline with system prompt + history + retrieval context; model `ChatTongyi` with tool binding (MCP db tools). Handles tool calls then regenerates final answer.
- Retrieval: `app/vector_db.py` defines `VectorDatabase` using DashScope embeddings + LanceDB (table `lance` in `./lance_db`) and BM25 fusion; `CustomStagedRetriever` returns merged results.
- MCP tools: `mcp_service/db_server.py` exposes `query_order_status`, `query_company_info` via `FastMCP` (stdio or optional HTTP) backed by MySQL pool in `db_pool.py`.
- Data/paths: LanceDB storage under `lance_db/`; MCP server path resolved relative to repo; MySQL config defaults host `localhost`, port `3306`, user/pass `root`/`root`, database `lgpt`.
- Key dependencies: langchain-core/community, langchain-mcp-adapters, DashScope (ChatTongyi & embeddings), LanceDB, jieba, mysql-connector, uvicorn, FastMCP, pydantic.