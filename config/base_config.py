# 基础配置
# config/base_config.py
class BaseConfig:
    # 模型配置
    model_name = "Qwen/Qwen2.5-14B"
    output_dir = "./web3_base_model"
    use_lora = True
    lora_r = 64
    lora_alpha = 256
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # 训练配置
    learning_rate = 2e-5
    train_batch_size = 4
    eval_batch_size = 4
    num_train_epochs = 3
    weight_decay = 0.01
    evaluation_strategy = "steps"
    eval_steps = 500
    save_strategy = "steps"
    save_steps = 500
    logging_dir = "./logs/base_training"
    logging_steps = 100
    fp16 = True
    bf16 = False
    gradient_accumulation_steps = 8
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"

    # 数据配置
    tokenizer_name = "Qwen/Qwen2.5-14B"
    max_length = 512

    # 分布式训练配置
    distributed_training = True