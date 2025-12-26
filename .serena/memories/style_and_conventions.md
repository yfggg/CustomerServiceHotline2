# Style & Conventions

- Language: Python 3; modules kept small; REPL driven CLI.
- Type hints: used for lists/Any; optional type hints on methods; Pydantic Field for dataclass-like configs.
- Imports: standard + third-party at top; relative imports for app modules.
- Docstrings/comments: brief Chinese docstrings for tool wrappers or clarifications; otherwise minimal comments.
- Prompt/content: Chinese system prompts; answers expected concise (<200 chars) with "依据" section; preserve formatting.
- Error handling: lightweight try/except around tool invocation and DB access; return text errors rather than raising.
- Config: API keys and DB creds currently hardcoded; no env loader; be cautious not to expose secrets further.
- Retrieval: BM25 uses jieba POS filter (`allowed_pos` set); keep allowed POS and weights configurable via params when extending.