import logging
import json
import logging
import os
from typing import Dict, List

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# 全局配置
# =========================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "LoRA/outputs/qwen2.5-7b-instruct-lora"
SYSTEM_PROMPT = "你是一个说话自然、不过度正式、像真人一样交流的中文助手。"
MAX_LENGTH = 256
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "style_samples.jsonl")
VALIDATION_PROMPTS = [
    "用更随意的语气跟我打声招呼，别像客服。",
    "我有点心烦，陪我聊两句就好。",
    "给我一句不鸡汤但很贴心的安慰。",
    "随便抛个轻松的话题。",
    "帮我用口语说一句：今天还不错。",
]

logging.basicConfig(level=logging.INFO)


# =========================
# 数据集（JSONL）
# =========================
def fallback_samples() -> List[Dict[str, str]]:
    return [
        {"user": "你是谁？", "assistant": "我是一个示例机器人，用来测试 LoRA 微调流程。"},
        {"user": "2+2 等于几？", "assistant": "2+2 等于 4，这是最基础的加法。"},
        {"user": "帮我随便打个招呼。", "assistant": "嗨，很高兴见到你，希望今天过得不错。"},
        {"user": "Python 怎么打印 Hello？", "assistant": "可以直接写 print('Hello')，然后运行脚本。"},
        {"user": "推荐一本科幻小说。", "assistant": "《三体》是个不错的选择，世界观非常宏大。"},
    ]


def load_style_samples(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        logging.warning("Style samples file not found: %s", path)
        return fallback_samples()

    examples: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Invalid JSONL at %s:%d", path, line_no)
                continue
            user = obj.get("user")
            assistant = obj.get("assistant")
            if not user or not assistant:
                continue
            examples.append({"user": user, "assistant": assistant})

    if not examples:
        logging.warning("Style samples empty, fallback to inline samples.")
        return fallback_samples()
    return examples


def build_toy_dataset() -> Dataset:
    examples = load_style_samples(SAMPLES_PATH)
    logging.info("Loaded %d samples from %s", len(examples), SAMPLES_PATH)
    return Dataset.from_list(examples)


# =========================
# SFT 核心：只对 Assistant 算 loss
# =========================
def tokenize_fn(batch, tokenizer):
    input_ids_list = []
    labels_list = []

    def common_prefix_len(a: List[int], b: List[int]) -> int:
        max_len = min(len(a), len(b))
        i = 0
        while i < max_len and a[i] == b[i]:
            i += 1
        return i

    def find_prefix_len(full_ids: List[int], prefix_ids: List[int]) -> int:
        if len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
            return len(prefix_ids)
        for i in range(1, len(full_ids) - len(prefix_ids) + 1):
            if full_ids[i : i + len(prefix_ids)] == prefix_ids:
                return i + len(prefix_ids)
        return common_prefix_len(full_ids, prefix_ids)

    for user_text, assistant_text in zip(batch["user"], batch["assistant"]):
        # 完整对话（system + user + assistant）
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        # prefix（system + user，用于 mask）
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )
        prefix_tokens = tokenizer(
            prefix_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()

        prefix_len = find_prefix_len(input_ids, prefix_tokens["input_ids"])
        labels[:prefix_len] = [-100] * prefix_len  # 只训 assistant

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }


# =========================
# 推理对比
# =========================
def run_generation(model, tokenizer, user_text: str, tag: str):
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    logging.info("[%s] %s", tag, text)


def run_validation_set(model, tokenizer, prompts: List[str], tag: str) -> None:
    for idx, prompt in enumerate(prompts, start=1):
        run_generation(model, tokenizer, prompt, f"{tag}-{idx}")


# =========================
# 主流程
# =========================
def main():
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_toy_dataset()
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # -------- QLoRA / 4bit --------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
    )
    model = prepare_model_for_kbit_training(model)

    # -------- LoRA 配置（Qwen / LLaMA）--------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------- 微调前对比 --------
    test_prompt = "用更像真人的语气，跟我打个招呼，别太正式。"
    run_generation(model, tokenizer, test_prompt, "微调前")
    run_validation_set(model, tokenizer, VALIDATION_PROMPTS, "泛化前")

    # -------- 训练 --------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # 更稳
        max_steps=200,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    def padding_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attention_mask = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=padding_collator,
    )

    trainer.train()

    # -------- 微调后对比 --------
    run_generation(model, tokenizer, test_prompt, "微调后")
    run_validation_set(model, tokenizer, VALIDATION_PROMPTS, "泛化后")

    # -------- 保存 LoRA --------
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("LoRA adapter 已保存到：%s", OUTPUT_DIR)
    adapter_model_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    adapter_config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(adapter_model_path) and os.path.exists(adapter_config_path):
        logging.info("Adapter 文件 OK: %s, %s", adapter_model_path, adapter_config_path)
    else:
        logging.warning("Adapter 文件缺失，请检查 %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
