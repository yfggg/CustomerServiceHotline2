import asyncio
from typing import Any, Dict, Iterable


class AlertRouter:
    """按渠道名称调用对应的 MCP 通知工具。"""

    def __init__(self, tools: Iterable[Any]) -> None:
        self.tool_map: Dict[str, Any] = {getattr(t, "name", ""): t for t in tools}

    def send(self, channels: Iterable[str], reason: str, question: str, ai_reply: str) -> None:
        payload = {"reason": reason, "question": question, "ai_reply": ai_reply}
        for channel in channels:
            tool_name = f"send_{channel}_alert"
            tool = self.tool_map.get(tool_name)
            if not tool:
                continue
            try:
                asyncio.run(tool.ainvoke(payload))
            except Exception:
                # 单通道失败不影响主流程，也不影响其他通道
                continue
