# 模型微调
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, PeftModel
import torch
import os
from datasets import concatenate_datasets


class Web3ModelFineTuner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        """加载基础模型并准备微调"""
        # 检查是加载基础模型还是微调后的模型
        if os.path.exists(os.path.join(self.config.base_model_path, "adapter_config.json")):
            # 加载已有的LoRA模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(
                base_model,
                self.config.base_model_path
            )
        else:
            # 加载基础模型并应用LoRA
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                inference_mode=False
            )

            model = get_peft_model(base_model, peft_config)

        # 打印可训练参数信息
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"Fine-tuning: trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

        return model

    def fine_tune_with_trading_data(self, train_dataset, eval_dataset):
        """基于交易数据微调模型"""
        model = self.setup_model()

        # 微调参数
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

        # 开始微调
        trainer.train()

        # 保存微调后的模型
        model.save_pretrained(self.config.output_dir)
        return model

    async def continuous_finetuning(self, data_stream):
        """基于实时数据流的持续微调"""
        model = self.setup_model()

        # 初始化优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000
        )

        # 持续接收数据流
        async for dataset_batch in data_stream.connect():
            # 准备数据
            tokenized_batch = self._tokenize_batch(dataset_batch)

            # 微调模型
            model.train()
            for epoch in range(self.config.online_epochs):
                for batch in self._get_data_loader(tokenized_batch):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}

                    # 前向传播
                    outputs = model(**inputs)
                    loss = outputs.loss

                    # 反向传播
                    loss.backward()

                    # 梯度累积和优化
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # 定期保存模型
            self._save_checkpoint(model)

    def _tokenize_batch(self, batch):
        """对数据批次进行tokenize"""
        # 这里需要根据实际数据格式实现tokenize逻辑
        # 简化示例，假设batch是一个包含text列的数据集
        tokenized = self.config.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return tokenized

    def _get_data_loader(self, tokenized_data, batch_size=4):
        """创建数据加载器"""
        # 简化示例，实际实现需要处理多种数据格式
        num_samples = len(tokenized_data["input_ids"])
        for i in range(0, num_samples, batch_size):
            batch = {
                k: v[i:i + batch_size] for k, v in tokenized_data.items()
            }
            yield batch

    def _save_checkpoint(self, model):
        """保存模型检查点"""
        # 创建带有时间戳的检查点目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"checkpoint-{timestamp}")

        # 保存模型
        model.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")