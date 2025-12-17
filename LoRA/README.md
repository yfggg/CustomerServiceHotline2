基于 Qwen2.5-7B-Instruct(Hugging Face 下载到本地服务器) 的 LoRA 训练脚本
监督微调, 你给它问题 + 标准答案，它学“这种问题我就该这么答”
SFT 题目的来源, 从真实客服 & 业务流程里拿

1️⃣ SFT / DPO / RLHF
👉 训练“范式 / 方法”（教什么、怎么教）
2️⃣ LoRA / QLoRA / 全参
👉 参数更新方式（模型权重怎么改）
3️⃣ 模型（Qwen2.5-7B-Instruct）
👉 被改的对象

Qwen2.5-7B-Instruct
  └── 用 LoRA 这种方式
        ├── 先做 SFT（监督微调）
        
阶段 1：SFT + LoRA
你在教模型：
“遇到这种财税问题，你应该这样判断、这样说。”
👉 改的是 LoRA adapter 权重

冻结模型，别急着再训
固定一套「评估问题集」
对比「SFT 前 vs SFT 后」

多 LoRA 并存（高级但好用）
比如：
发票 LoRA
个税 LoRA
企税 LoRA