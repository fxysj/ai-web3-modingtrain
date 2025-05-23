# 微调模型
# scripts/finetune_model.py
import os
import argparse
import torch
import json
import time
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
from utils.logging_utils import setup_logger
from utils.metrics_utils import compute_metrics
from data.data_preprocessor import Web3DataPreprocessor

# 设置日志
logger = setup_logger("finetune_model")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Finetune a Web3.0 model")

    # 基础模型配置
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-14B",
                        help="基础模型路径或名称")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径，默认为基础模型路径")

    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default="0xscope/web3-trading-analysis",
                        help="用于微调的数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="本地数据集目录")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")

    # LoRA配置
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用LoRA进行微调")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout率")
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj"],
                        help="LoRA目标模块")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="./web3_finetuned_model",
                        help="模型输出目录")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="学习率预热比例")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估频率（步数）")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="保存策略")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存频率（步数）")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="日志记录频率（步数）")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="是否使用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="是否使用BF16混合精度")
    parser.add_argument("--tf32", action="store_true", default=False,
                        help="是否使用TF32精度（NVIDIA A100及以上）")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True,
                        help="训练结束后是否加载最佳模型")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                        help="用于选择最佳模型的指标")

    # 其他配置
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练本地排名")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="调试模式")

    return parser.parse_args()


def load_and_preprocess_data(args, tokenizer):
    """加载并预处理数据集"""
    logger.info(f"Loading dataset: {args.dataset_name}")

    # 加载数据集
    if args.data_dir:
        # 从本地目录加载
        dataset = load_dataset("json", data_dir=args.data_dir)
    else:
        # 从Hugging Face Hub加载
        dataset = load_dataset(args.dataset_name)

    # 确保数据集包含所需的字段
    required_fields = ["question", "answer"]
    for split in dataset:
        if not all(field in dataset[split].column_names for field in required_fields):
            raise ValueError(f"Dataset split {split} missing required fields: {required_fields}")

    # 预处理数据
    preprocessor = Web3DataPreprocessor(tokenizer, max_length=args.max_seq_length)
    processed_dataset = dataset.map(
        preprocessor.process_example,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4 if not args.debug else 1
    )

    logger.info(f"Dataset loaded and processed: {processed_dataset}")
    return processed_dataset


def setup_model(args):
    """设置模型和LoRA配置"""
    logger.info(f"Loading base model: {args.base_model_path}")

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16 if args.fp16 or args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.base_model_path,
        trust_remote_code=True,
    )

    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 如果使用LoRA
    if args.use_lora:
        logger.info("Setting up LoRA configuration")

        # 准备模型进行LoRA训练
        if args.fp16 or args.bf16:
            model = prepare_model_for_kbit_training(model)

        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )

        # 应用LoRA配置
        model = get_peft_model(model, lora_config)

        # 打印可训练参数
        model.print_trainable_parameters()

    return model, tokenizer


def run_finetuning(args, model, tokenizer, dataset):
    """运行模型微调"""
    logger.info("Setting up training arguments")

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        report_to=[],  # 不使用任何报告工具
        seed=args.seed,
        dataloader_num_workers=4 if not args.debug else 0,
        remove_unused_columns=False,
    )

    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    logger.info("Initializing Trainer")

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics if not args.debug else None,
    )

    logger.info("Starting fine-tuning")

    # 开始微调
    train_result = trainer.train()

    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # 保存分词器
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练参数
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)

    logger.info("Fine-tuning completed successfully")
    return model


def main():
    """主函数"""
    args = parse_args()

    # 打印参数
    logger.info(f"Fine-tuning arguments: {args}")

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 设置模型和分词器
    model, tokenizer = setup_model(args)

    # 加载并预处理数据
    dataset = load_and_preprocess_data(args, tokenizer)

    # 运行微调
    model = run_finetuning(args, model, tokenizer, dataset)

    # 测试微调后的模型
    if not args.debug:
        logger.info("Testing the fine-tuned model")

        # 准备测试提示
        test_prompts = [
            "分析以太坊链上最近一周的交易趋势，特别是DeFi协议的表现。",
            "预测比特币价格在未来一个月的走势，并分析主要影响因素。",
            "解释Uniswap V3与V2相比的主要创新点和优势。",
        ]

        # 生成回答
        model.eval()
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_seq_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            logger.info("-" * 80)


if __name__ == "__main__":
    main()