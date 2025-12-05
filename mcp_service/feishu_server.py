import json
import sys
from datetime import datetime
from typing import List, Optional
from urllib import error, request
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

app = FastMCP("mcp-feishu-server")


def _post_webhook(webhook: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=5) as resp:
        resp.read()


@app.tool()
async def send_feishu_alert(
    reason: str,
    question: str,
    ai_reply: str,
    history_lines: Optional[List[str]] = None,
    tool_outputs: Optional[List[str]] = None,
) -> TextContent:
    """
    发送飞书群机器人通知，便于人工接管。
    环境变量 FEISHU_WEBHOOK 需要配置为飞书群自定义机器人地址。
    """
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/149438e3-b275-47c1-b3f6-da89906b06b9"

    history_text = "\n".join(history_lines or [])
    tool_text = "\n".join(tool_outputs or [])

    payload = {
        "msg_type": "text",
        "content": {
            "text": (
                f"原因: {reason}\n"
                f"用户问题: {question}\n"
                f"AI回复: {ai_reply}\n"
                f"工具: {tool_text or '-'}\n"
                f"历史: {history_text or '-'}\n"
                f"时间: {datetime.now().isoformat(timespec='seconds')}"
            )
        },
    }

    try:
        _post_webhook(webhook, payload)
        return TextContent(type="text", text="飞书通知已发送")
    except error.URLError as exc:
        return TextContent(type="text", text=f"飞书通知失败: {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        import uvicorn  # type: ignore

        uvicorn.run(app.streamable_http_app(), host="127.0.0.1", port=8001, log_level="info")
    else:
        print("Feishu MCP Server running in stdio mode...", file=sys.stderr)
        import asyncio

        asyncio.run(app.run_stdio_async())
