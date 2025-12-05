import asyncio
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.vector_db import CustomStagedRetriever


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 系统消息
system_message = (
    "1. 您是AI客服助手，回复简短自然（≤200字），像真人客服。\n"
    "2. 必须结合上下文和历史作答。\n"
    "3. 信息缺失或模糊时，先一句澄清，再给已有依据下的建议，避免模板化套话。\n"
    "4. 超出能力范围时说明需人工支持，并给出下一步。\n"
    "5. 对未知信息注明依据不足；无效输入提示用户重新表述。\n"
    "6. 回复格式：正文（可含确认或指导）+ “依据：”后简述依据来自历史或检索结果。\n"
)


class ChatAssistant:
    def __init__(self):
        self.messages: List[Any] = []
        self.tools = asyncio.run(self._load_mcp_tools())
        self.chain = self._build_chain()

    @staticmethod
    async def _load_mcp_tools():
        """启动本地 MCP db 服务（stdio），并获取工具列表。"""
        server_path = Path(__file__).resolve().parent.parent / "mcp_service" / "db_server.py"
        client = MultiServerMCPClient(
            {
                "db": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(server_path)],
                    "cwd": str(server_path.parent),
                }
            }
        )
        tools = await client.get_tools()
        # 不主动关闭，保持会话存活；langchain-mcp-adapters 当前无 aclose/close
        return tools

    def _build_chain(self):
        retriever = CustomStagedRetriever()

        build_inputs = RunnableMap(
            {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(lambda q: _format_docs(retriever.invoke(q))),
                "system_message": RunnableLambda(lambda q: system_message),
                "chat_history": RunnableLambda(lambda q: self._get_filtered_history()),
            }
        )

        enhance_question = RunnableLambda(lambda data: {**data, "question": f"{data['question']}"})

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "[检索结果]\n{context}\n"),
                ("human", "{question}"),
            ]
        )

        model = ChatTongyi(
            model="qwen-plus-2025-09-11",
            dashscope_api_key="sk-b2817c33fdd64d3189582e100e1c0617",
            temperature=0.1,
            streaming=False,
        ).bind_tools(self.tools)

        return build_inputs | enhance_question | prompt | model

    def _get_filtered_history(self) -> List[Any]:
        history_messages: List[Any] = self.messages
        if history_messages and isinstance(history_messages[-1], HumanMessage):
            history_messages = history_messages[:-1]

        trimmed = trim_messages(
            history_messages,
            token_counter=lambda m: len(getattr(m, "content", "")),
            max_tokens=1600,
            strategy="last",
            include_system=False,
        )
        return trimmed or []

    def _handle_tool_calls(self, result_msg: AIMessage) -> List[str]:
        """执行工具，写入历史，返回所有工具结果列表。"""
        tool_messages: List[ToolMessage] = []
        combined_results: List[str] = []

        tool_map = {t.name: t for t in self.tools}
        for call in result_msg.tool_calls:
            tool_name = call["name"]
            tool_args: dict[str, Any] = call["args"]
            tool_call_id = call.get("id", "")

            if tool_name not in tool_map:
                tool_result = f"未知工具：{tool_name}"
            else:
                try:
                    tool_result = asyncio.run(tool_map[tool_name].ainvoke(tool_args))
                except Exception as exc:
                    tool_result = f"工具执行失败：{exc}"

            combined_results.append(str(tool_result))
            tool_messages.append(
                ToolMessage(content=str(tool_result), name=tool_name, tool_call_id=tool_call_id,)
            )

        self.messages.append(result_msg)
        self.messages.extend(tool_messages)
        return combined_results

    def stream_chat(self, question: str) -> str:
        cleaned_question = question.strip()
        if not cleaned_question:
            return ""

        self.messages.append(HumanMessage(content=cleaned_question))

        try:
            result_msg: AIMessage = self.chain.invoke(cleaned_question)

            if getattr(result_msg, "tool_calls", None):
                # 执行工具并写入历史
                self._handle_tool_calls(result_msg)
                # 再次调用模型，基于完整历史（含工具结果）生成综合回复
                final_msg: AIMessage = self.chain.invoke(cleaned_question)
                self.messages.append(final_msg)
                return final_msg.content

            self.messages.append(result_msg)
            return result_msg.content
        except Exception as e:
            return str(e)


# if __name__ == "__main__":
#     assistant = ChatAssistant()
#     while True:
#         q = input("你: ").strip()
#         if not q:
#             continue
#         if q.lower() in {"exit", "quit", "q"}:
#             break
#         print("助手:", assistant.stream_chat(q))
