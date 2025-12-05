import json
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage


_DECISION_PROMPT = """你是客服升级决策器，只输出 JSON。
规则：
- 应升级：用户明确要转人工/投诉/报障；表达强烈不满；AI回复无效或反复未解决；涉及高风险修改（如支付/手机号/身份）。
- 不升级：一般咨询已解答；用户接受答案；纯闲聊。
输出格式：{{"escalate": true|false, "reason": "简要原因", "channels": ["feishu"]}}
示例1：
用户: 我要投诉，人工立刻来
AI: 抱歉给您带来不便，我们会协助
输出: {{"escalate": true, "reason": "用户投诉并要求人工", "channels": ["feishu"]}}
示例2：
用户: 你们客服电话多少？
AI: 400-123-456
输出: {{"escalate": false, "reason": "一般咨询已解答", "channels": ["feishu"]}}
当前输入：
用户: {question}
AI: {ai_reply}
仅输出 JSON，不要任何解释。"""


@dataclass
class AlertDecision:
    escalate: bool
    reason: str
    channels: List[str]


class AlertDecisionAgent:
    """调用 LLM 做人工升级决策。"""

    def __init__(self, llm: Optional[Any] = None) -> None:
        if llm is None:
            api_key = os.getenv("DASHSCOPE_API_KEY", "sk-b2817c33fdd64d3189582e100e1c0617")
            self.llm = ChatTongyi(
                model="qwen-plus-2025-09-11",
                dashscope_api_key=api_key,
                temperature=0.1,
                streaming=False,
            )
        else:
            self.llm = llm

    def decide(self, question: str, ai_reply: str) -> AlertDecision:
        prompt = _DECISION_PROMPT.format(question=question, ai_reply=ai_reply)

        try:
            result = self.llm.invoke(prompt)
            content = result.content if isinstance(result, AIMessage) else getattr(result, "content", str(result))
            data = json.loads(str(content))
            escalate = bool(data.get("escalate"))
            reason = str(data.get("reason") or "").strip() or "未提供原因"
            channels_raw = data.get("channels", ["feishu"])
            channels: List[str] = []
            if isinstance(channels_raw, list):
                channels = [str(c) for c in channels_raw if str(c).strip()]
            if not channels:
                channels = ["feishu"]
            return AlertDecision(escalate=escalate, reason=reason, channels=channels)
        except Exception:
            return AlertDecision(escalate=False, reason="决策解析失败", channels=["feishu"])
