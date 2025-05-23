# 微调配置
# utils/fine_tuning_config.py
import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)


@dataclass
class ModelArguments:
    """
    模型相关参数配置
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练模型名称或路径"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "配置名称，用于加载模型配置"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "分词器名称，用于加载分词器"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "缓存目录，用于存储预训练模型"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "是否使用快速分词器"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "模型版本"}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "是否使用认证令牌"}
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None, metadata={"help": "是否调整位置嵌入大小"}
    )
    web3_specific_initialization: bool = field(
        default=False, metadata={"help": "是否使用Web3.0特定初始化"}
    )


@dataclass
class DataTrainingArguments:
    """
    数据和训练相关参数配置
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "数据集名称"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集配置名称"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "训练数据文件路径"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "验证数据文件路径"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "测试数据文件路径"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖缓存"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "预处理工作线程数"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "输入序列的最大长度。当超过这个长度时，序列将被截断。"
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "是否将所有样本填充到max_seq_length。"
                    "如果设置为False，将使用动态填充。"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "限制训练样本数量，用于调试或快速实验"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "限制评估样本数量，用于调试或快速实验"
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "限制预测样本数量，用于调试或快速实验"
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "标签平滑因子"}
    )
    web3_data_augmentation: bool = field(
        default=False, metadata={"help": "是否使用Web3.0数据增强"}
    )
    web3_specific_metrics: bool = field(
        default=True, metadata={"help": "是否使用Web3.0特定评估指标"}
    )


@dataclass
class Web3TrainingArguments(TrainingArguments):
    """
    自定义Web3.0训练参数
    """
    optimizer: str = field(
        default="adamw",
        metadata={"help": "优化器类型 (adamw, adafactor, lamb)"}
    )
    learning_rate_scheduler: str = field(
        default="linear",
        metadata={"help": "学习率调度器类型 (linear, cosine, constant)"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "学习率热身比例"}
    )
    weight_decay_groups: str = field(
        default="all",
        metadata={"help": "应用权重衰减的参数组 (all, none, no_bias)"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "是否使用梯度检查点以节省内存"}
    )
    use_fp16: bool = field(
        default=False,
        metadata={"help": "是否使用半精度浮点数 (FP16) 训练"}
    )
    use_bf16: bool = field(
        default=False,
        metadata={"help": "是否使用Brain Floating Point (BF16) 训练"}
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "早停耐心值，即多少个评估周期没有改善后停止训练"}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "早停阈值，即评估指标需要提高的最小幅度"}
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "日志记录频率"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "模型保存策略 (steps, epoch)"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "模型保存频率"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "评估策略 (no, steps, epoch)"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估频率"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "训练结束后是否加载最佳模型"}
    )
    metric_for_best_model: str = field(
        default="loss",
        metadata={"help": "用于选择最佳模型的指标"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "metric_for_best_model是否越大越好"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "报告训练结果的工具列表"}
    )


def get_optimizer(model: torch.nn.Module, args: Web3TrainingArguments) -> torch.optim.Optimizer:
    """
    根据配置获取优化器

    Args:
        model: 模型
        args: 训练参数

    Returns:
        优化器
    """
    # 根据weight_decay_groups配置权重衰减
    if args.weight_decay_groups == "all":
        # 对所有参数应用权重衰减
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
        ]
    elif args.weight_decay_groups == "none":
        # 不对任何参数应用权重衰减
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad], "weight_decay": 0.0}
        ]
    elif args.weight_decay_groups == "no_bias":
        # 不对偏置和LayerNorm参数应用权重衰减
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
    else:
        raise ValueError(f"Unsupported weight_decay_groups: {args.weight_decay_groups}")

    # 根据optimizer参数选择优化器
    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay if args.weight_decay_groups != "none" else 0.0,
        )
    elif args.optimizer.lower() == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=args.weight_decay if args.weight_decay_groups != "none" else 0.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "lamb":
        from transformers.optimization import Lamb
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay if args.weight_decay_groups != "none" else 0.0,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, args: Web3TrainingArguments,
                  num_training_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    根据配置获取学习率调度器

    Args:
        optimizer: 优化器
        args: 训练参数
        num_training_steps: 训练总步数

    Returns:
        学习率调度器
    """
    # 计算热身步数
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    # 根据learning_rate_scheduler参数选择调度器
    if args.learning_rate_scheduler.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.learning_rate_scheduler.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
    elif args.learning_rate_scheduler.lower() == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
    else:
        raise ValueError(f"Unsupported learning_rate_scheduler: {args.learning_rate_scheduler}")

    return scheduler


def save_config(args: Union[ModelArguments, DataTrainingArguments, Web3TrainingArguments],
                output_dir: str, config_name: str = "config.json") -> None:
    """
    保存配置到JSON文件

    Args:
        args: 配置参数
        output_dir: 输出目录
        config_name: 配置文件名
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=2)


def load_config(config_path: str, config_class: type) -> object:
    """
    从JSON文件加载配置

    Args:
        config_path: 配置文件路径
        config_class: 配置类

    Returns:
        配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 创建配置对象
    config = config_class(**config_dict)

    return config


def get_default_web3_config() -> Tuple[ModelArguments, DataTrainingArguments, Web3TrainingArguments]:
    """
    获取Web3.0领域的默认配置

    Returns:
        模型参数、数据训练参数和Web3训练参数的元组
    """
    # 解析默认参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Web3TrainingArguments))

    # 默认参数
    model_args = ModelArguments(
        model_name_or_path="bert-base-chinese",
        web3_specific_initialization=True
    )

    data_args = DataTrainingArguments(
        max_seq_length=512,
        pad_to_max_length=True,
        web3_data_augmentation=True,
        web3_specific_metrics=True
    )

    training_args = Web3TrainingArguments(
        output_dir="./web3_model",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        weight_decay_groups="no_bias",
        optimizer="adamw",
        learning_rate_scheduler="linear",
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="rouge-l",
        greater_is_better=True,
        logging_dir="./logs",
        report_to=["tensorboard"]
    )

    return model_args, data_args, training_args


# 示例使用
if __name__ == "__main__":
    # 获取默认Web3.0配置
    model_args, data_args, training_args = get_default_web3_config()

    # 打印配置信息
    print("模型配置:")
    for key, value in model_args.__dict__.items():
        print(f"  {key}: {value}")

    print("\n数据训练配置:")
    for key, value in data_args.__dict__.items():
        print(f"  {key}: {value}")

    print("\nWeb3训练配置:")
    for key, value in training_args.__dict__.items():
        print(f"  {key}: {value}")

    # 保存配置到文件
    save_config(model_args, "./configs", "model_config.json")
    save_config(data_args, "./configs", "data_config.json")
    save_config(training_args, "./configs", "training_config.json")

    # 从文件加载配置
    loaded_model_args = load_config("./configs/model_config.json", ModelArguments)
    print(f"\n加载的模型配置 - model_name_or_path: {loaded_model_args.model_name_or_path}")