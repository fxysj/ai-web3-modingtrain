# 训练基础模型
# scripts/train_base_model.py
import os
import argparse
import json
import torch
import logging
import math
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils.logging_utils import setup_logger
from utils.metrics_utils import compute_metrics
from data.data_preprocessor import Web3DataPreprocessor

# 设置日志
logger = setup_logger("train_base_model")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train a base Web3.0 model")

    # 基础模型配置
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-14B",
                        help="预训练模型名称或路径")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="分词器名称，默认为模型名称")

    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default="0xscope/web3-trading-analysis",
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="本地数据集目录")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--validation_split_percentage", type=int, default=5,
                        help="从训练集划分验证集的比例")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="./web3_base_model",
                        help="模型输出目录")
    parser.add_argument("--do_train", action="store_true", default=True,
                        help="是否进行训练")
    parser.add_argument("--do_eval", action="store_true", default=True,
                        help="是否进行评估")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减率")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"],
                        help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="学习率预热比例")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["steps", "epoch"],
                        help="模型保存策略")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="是否使用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="是否使用BF16混合精度")
    parser.add_argument("--tf32", action="store_true", default=False,
                        help="是否使用TF32 (需要A100或H100 GPU)")

    # LoRA配置
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="是否使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout率")
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj"],
                        help="LoRA目标模块")

    # 其他配置
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="报告工具 (none/tensorboard/wandb)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="运行名称")

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

    # 如果没有验证集，从训练集划分
    if "validation" not in dataset:
        logger.info(f"Creating validation set from {args.validation_split_percentage}% of training data")
        dataset = dataset["train"].train_test_split(
            test_size=args.validation_split_percentage / 100,
            seed=args.seed
        )
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["test"]
        })

    # 预处理数据
    preprocessor = Web3DataPreprocessor(tokenizer, max_length=args.max_seq_length)
    processed_dataset = dataset.map(
        preprocessor.process_example,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    logger.info(f"Dataset loaded and processed: {processed_dataset}")
    return processed_dataset


def setup_model_and_tokenizer(args):
    """设置模型和分词器"""
    logger.info(f"Loading model: {args.model_name_or_path}")

    # 加载分词器
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )

    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.device_count() > 1 else None
    )

    # 如果使用LoRA
    if args.use_lora:
        logger.info("Configuring LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存参数配置
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 初始化加速器
    accelerator = Accelerator(
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to if args.report_to != "none" else None,
        project_dir=args.output_dir
    )

    # 设置运行名称
    if not args.run_name:
        args.run_name = f"web3-base-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer(args)

    # 加载和预处理数据
    dataset = load_and_preprocess_data(args, tokenizer)

    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 对于因果语言模型，不需要掩码语言模型任务
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # 准备加速
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 开始训练
    if accelerator.is_main_process:
        accelerator.init_trackers(args.run_name, config=vars(args))
        logger.info("***** 开始训练 *****")
        logger.info(f"  训练样本数 = {len(dataset['train'])}")
        logger.info(f"  验证样本数 = {len(dataset['validation'])}")
        logger.info(f"  每个设备的训练批次大小 = {args.per_device_train_batch_size}")
        logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
        logger.info(
            f"  总训练批次大小 = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"  训练轮数 = {args.num_train_epochs}")
        logger.info(f"  总训练步数 = {max_train_steps}")
        logger.info(f"  预热步数 = {num_warmup_steps}")

    # 训练循环
    completed_steps = 0
    best_eval_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=not accelerator.is_main_process
        )

        for step, batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 日志记录
            if step % args.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}, LR: {lr:.8f}, Loss: {avg_loss:.4f}")

                if accelerator.is_main_process:
                    accelerator.log({
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "epoch": epoch + (step + 1) / len(train_dataloader)
                    }, step=completed_steps)

            completed_steps += 1

            # 评估和保存
            if (args.save_strategy == "steps" and completed_steps % args.save_steps == 0) or \
                    (completed_steps == max_train_steps):
                # 评估
                if args.do_eval:
                    model.eval()
                    eval_loss = 0
                    eval_steps = 0

                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            loss = outputs.loss
                        eval_loss += loss.detach().float()
                        eval_steps += 1

                    avg_eval_loss = eval_loss / eval_steps

                    if accelerator.is_main_process:
                        logger.info(f"Step {completed_steps}: Evaluation loss = {avg_eval_loss:.4f}")
                        accelerator.log({"eval_loss": avg_eval_loss}, step=completed_steps)

                        # 保存最佳模型
                        if avg_eval_loss < best_eval_loss:
                            best_eval_loss = avg_eval_loss
                            save_dir = os.path.join(args.output_dir, "best_model")
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(save_dir)
                                logger.info(f"Best model saved to {save_dir} with eval loss: {best_eval_loss:.4f}")

                # 保存当前模型
                if accelerator.is_main_process:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(save_dir)
                    logger.info(f"Model checkpoint saved to {save_dir}")

        # 每轮结束后的评估和保存
        if args.save_strategy == "epoch":
            if args.do_eval:
                model.eval()
                eval_loss = 0
                eval_steps = 0

                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                        loss = outputs.loss
                    eval_loss += loss.detach().float()
                    eval_steps += 1

                avg_eval_loss = eval_loss / eval_steps

                if accelerator.is_main_process:
                    logger.info(f"Epoch {epoch + 1}: Evaluation loss = {avg_eval_loss:.4f}")
                    accelerator.log({"eval_loss": avg_eval_loss}, step=completed_steps)

                    # 保存最佳模型
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        save_dir = os.path.join(args.output_dir, "best_model")
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(save_dir)
                            logger.info(f"Best model saved to {save_dir} with eval loss: {best_eval_loss:.4f}")

    # 保存最终模型
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final_model")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Final model saved to {save_dir}")

        # 结束跟踪
        accelerator.end_training()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()