import asyncio
import sys
import uvicorn
from pathlib import Path
from typing import Any, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# 兼容包内/脚本直接运行的导入
try:
    from mcp_db_service.db_pool import db_pool
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from db_pool import db_pool  # type: ignore

app = FastMCP("mcp-db-server")


@app.tool()
async def query_order_status(order_id: str) -> TextContent:
    """按订单号查询订单状态/明细"""
    if not order_id:
        return TextContent(type="text", text="缺少订单号")

    sql = "SELECT public_id, state, update_time, remarks FROM pay_order WHERE public_id = %s LIMIT 1"
    row = db_pool.fetch_one(sql, (order_id,))
    if not row:
        return TextContent(type="text", text="未找到该订单，请核对订单号")

    return TextContent(
        type="text",
        text=(
            f"订单 {row.get('public_id', order_id)} 状态：{row.get('state', '未知')}，"
            f"更新时间：{row.get('update_time', '未知')}；备注：{row.get('remarks') or '无'}"
        ),
    )


@app.tool()
async def query_company_info(company_name: Optional[str] = None, tax_id: Optional[str] = None) -> TextContent:
    """按公司全称或统一社会信用代码查询公司信息"""
    name = (company_name or "").strip()
    tax = (tax_id or "").strip()
    if not name and not tax:
        return TextContent(type="text", text="缺少公司名称或统一社会信用代码")

    conditions: List[str] = []
    params: List[Any] = []
    if name:
        conditions.append("name = %s")
        params.append(name)
    if tax:
        conditions.append("tax_id = %s")
        params.append(tax)

    where_clause = " OR ".join(conditions)
    sql = f"SELECT name, tax_id, state FROM company WHERE {where_clause} LIMIT 5"
    rows = db_pool.fetch_all(sql, params)
    if not rows:
        return TextContent(type="text", text="未找到公司信息，请核对公司全称或统一社会信用代码")

    if len(rows) == 1:
        top = rows[0]
        return TextContent(
            type="text",
            text=(
                f"公司：{top.get('name')}；"
                f"税号：{top.get('tax_id')}；"
                f"状态：{top.get('state', '未知')}"
            ),
        )

    lines = [
        f"{idx + 1}. 公司：{r.get('name')}；税号：{r.get('tax_id')}；状态：{r.get('state', '未知')}"
        for idx, r in enumerate(rows)
    ]
    return TextContent(type="text", text="找到多条记录：\n" + "\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        uvicorn.run(app.streamable_http_app(), host="127.0.0.1", port=8000, log_level="info")
    else:
        print("MCP DB Server running in stdio mode...", file=sys.stderr)
        asyncio.run(app.run_stdio_async())
