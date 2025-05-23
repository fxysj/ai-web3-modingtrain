# 数据集加载与预处理
# data/dataset_loader.py
from datasets import load_dataset, DatasetDict
import torch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset
import json


class Web3DatasetLoader:
    def __init__(self, tokenizer_name="Qwen/Qwen2.5-14B", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_base_dataset(self, dataset_name="mamung/x_dataset_192", split_ratio=[0.8, 0.1, 0.1]):
        """加载基础Web3.0数据集并划分训练/验证/测试集"""
        try:
            dataset = load_dataset(dataset_name)
        except:
            # 如果直接加载失败，尝试手动处理
            dataset = self._load_and_process_dataset(dataset_name)

        # 确保有训练集和验证集
        if "train" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=split_ratio[2])

        if "validation" not in dataset:
            train_val = dataset["train"].train_test_split(test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]))
            dataset = DatasetDict({
                "train": train_val["train"],
                "validation": train_val["test"],
                "test": dataset["test"] if "test" in dataset else None
            })

        return self._tokenize_dataset(dataset)

    def load_trading_dataset(self, dataset_name="0xscope/web3-trading-analysis", split_ratio=[0.8, 0.1, 0.1]):
        """加载交易数据集用于微调"""
        try:
            dataset = load_dataset(dataset_name)
        except:
            dataset = self._load_and_process_dataset(dataset_name)

        # 处理大型交易数据集
        if "train" in dataset and len(dataset["train"]) > 1000000:
            # 对于超大规模数据集，采样一部分用于快速验证
            small_train = dataset["train"].select(range(1000000))
            dataset["train"] = small_train

        return self._tokenize_dataset(dataset)

    def _load_and_process_dataset(self, dataset_path):
        """手动加载和处理数据集"""
        try:
            # 尝试作为JSONL文件加载
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        except:
            # 尝试作为CSV文件加载
            df = pd.read_csv(dataset_path)

        # 确保数据集中有"text"列
        if "text" not in df.columns:
            # 尝试组合其他列
            if "question" in df.columns and "answer" in df.columns:
                df["text"] = df["question"] + " " + df["answer"]
            else:
                raise ValueError("Dataset does not contain 'text' column or suitable alternatives.")

        # 创建数据集
        dataset = Dataset.from_pandas(df)
        return dataset

    def _tokenize_dataset(self, dataset):
        """对数据集进行tokenize处理"""

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

        # 只处理包含text列的数据集
        columns_to_remove = []
        if "text" in dataset.column_names.get("train", []):
            columns_to_remove = ["text"]

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

        # 确保有labels列
        if "labels" not in tokenized_datasets.column_names.get("train", []):
            if "label" in tokenized_datasets.column_names.get("train", []):
                tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            else:
                # 如果没有标签，创建一个虚拟标签
                tokenized_datasets = tokenized_datasets.add_column("labels", [0] * len(tokenized_datasets["train"]))

        return tokenized_datasets


