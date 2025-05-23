# 基础模型训练
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os


class Web3ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        """设置基础模型和LoRA配置"""
        # 分布式训练设置
        if self.config.distributed_training:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

            # 使用accelerate加载模型
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )

            # 使用FSDP进行分布式训练
            model = load_checkpoint_and_dispatch(
                model,
                self.config.model_name,
                device_map="auto",
                dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.device_count() > 1 else "auto"
            )

        # LoRA配置
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                inference_mode=False
            )
            model = get_peft_model(model, peft_config)

        # 打印可训练参数信息
        self._print_trainable_parameters(model)

        return model

    def _print_trainable_parameters(self, model):
        """打印可训练参数信息"""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def train_base_model(self, train_dataset, eval_dataset):
        """训练基础Web3.0模型"""
        model = self.setup_model()

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            push_to_hub=False,
            remove_unused_columns=False,
            report_to=[]  # 不使用任何报告工具
        )

        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        model.save_pretrained(self.config.output_dir)
        return model