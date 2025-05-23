# 模型评估
# models/model_evaluator.py
import torch
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


class Web3ModelEvaluator:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            device: torch.device = None,
            eval_dataset: Optional[Dict[str, Any]] = None,
            output_dir: str = "./evaluation_results",
    ):
        """
        初始化模型评估器

        Args:
            model: 要评估的模型
            tokenizer: 模型使用的分词器
            device: 评估使用的设备
            eval_dataset: 评估数据集
            output_dir: 评估结果保存目录
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

        # 创建保存目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化评估指标
        self.metrics = {
            "accuracy": load_metric("accuracy"),
            "f1": load_metric("f1"),
            "precision": load_metric("precision"),
            "recall": load_metric("recall"),
            "rouge": load_metric("rouge"),
            "bleu": load_metric("bleu"),
            "perplexity": load_metric("perplexity"),
        }

    def evaluate(self, dataset: Optional[Dict[str, Any]] = None, batch_size: int = 8) -> Dict[str, float]:
        """
        对模型进行全面评估

        Args:
            dataset: 要评估的数据集，如果为None则使用初始化时的数据集
            batch_size: 评估批次大小

        Returns:
            包含各种评估指标的字典
        """
        if dataset is None:
            dataset = self.eval_dataset

        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        results = {}

        # 评估生成任务
        if "text" in dataset and "target" in dataset:
            print("Evaluating text generation...")
            generation_results = self.evaluate_text_generation(
                dataset["text"], dataset["target"], batch_size=batch_size
            )
            results.update(generation_results)

        # 评估分类任务
        if "input_ids" in dataset and "labels" in dataset:
            print("Evaluating classification...")
            classification_results = self.evaluate_classification(
                dataset["input_ids"], dataset["labels"], batch_size=batch_size
            )
            results.update(classification_results)

        # 计算困惑度
        if "input_ids" in dataset:
            print("Calculating perplexity...")
            perplexity = self.calculate_perplexity(dataset["input_ids"], batch_size=batch_size)
            results["perplexity"] = perplexity

        # 保存评估结果
        self._save_results(results)

        return results

    def evaluate_text_generation(
            self,
            prompts: List[str],
            targets: List[str],
            batch_size: int = 8,
            max_length: int = 512,
            num_beams: int = 4
    ) -> Dict[str, float]:
        """
        评估文本生成任务

        Args:
            prompts: 输入提示列表
            targets: 目标文本列表
            batch_size: 批次大小
            max_length: 生成的最大长度
            num_beams: 束搜索宽度

        Returns:
            包含ROUGE和BLEU等指标的字典
        """
        self.model.eval()

        predictions = []

        # 分批处理
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]

            # 编码输入
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="torch",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # 解码生成的文本
            batch_predictions = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            predictions.extend(batch_predictions)

        # 计算评估指标
        rouge_scores = self.metrics["rouge"].compute(
            predictions=predictions, references=targets
        )

        # 提取主要ROUGE指标
        results = {
            "rouge1": rouge_scores["rouge1"].mid.fmeasure,
            "rouge2": rouge_scores["rouge2"].mid.fmeasure,
            "rougeL": rouge_scores["rougeL"].mid.fmeasure,
        }

        # 计算BLEU分数（需要分词后的结果）
        bleu_predictions = [prediction.split() for prediction in predictions]
        bleu_references = [[target.split()] for target in targets]

        try:
            bleu_score = self.metrics["bleu"].compute(
                predictions=bleu_predictions, references=bleu_references
            )
            results["bleu"] = bleu_score["bleu"]
        except Exception as e:
            print(f"Error computing BLEU score: {e}")
            results["bleu"] = 0.0

        # 保存一些样本用于人工检查
        self._save_generation_samples(prompts, predictions, targets)

        return results

    def evaluate_classification(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            batch_size: int = 8
    ) -> Dict[str, float]:
        """
        评估分类任务

        Args:
            input_ids: 输入token ID
            labels: 标签
            batch_size: 批次大小

        Returns:
            包含准确率、F1分数等指标的字典
        """
        self.model.eval()

        all_predictions = []
        all_labels = []

        # 分批处理
        for i in tqdm(range(0, len(input_ids), batch_size)):
            batch_input_ids = input_ids[i:i + batch_size].to(self.device)
            batch_labels = labels[i:i + batch_size].to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.model(batch_input_ids)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

        # 计算评估指标
        accuracy = self.metrics["accuracy"].compute(
            predictions=all_predictions, references=all_labels
        )["accuracy"]

        f1 = self.metrics["f1"].compute(
            predictions=all_predictions, references=all_labels, average="weighted"
        )["f1"]

        precision = self.metrics["precision"].compute(
            predictions=all_predictions, references=all_labels, average="weighted"
        )["precision"]

        recall = self.metrics["recall"].compute(
            predictions=all_predictions, references=all_labels, average="weighted"
        )["recall"]

        results = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        # 绘制混淆矩阵
        self._plot_confusion_matrix(all_labels, all_predictions)

        return results

    def calculate_perplexity(self, input_ids: torch.Tensor, batch_size: int = 8) -> float:
        """
        计算模型的困惑度

        Args:
            input_ids: 输入token ID
            batch_size: 批次大小

        Returns:
            困惑度值
        """
        self.model.eval()

        # 准备输入数据
        texts = []
        for ids in input_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)

        # 计算困惑度
        perplexity = self.metrics["perplexity"].compute(
            model_id=self.model,
            texts=texts,
            batch_size=batch_size,
            device=str(self.device)
        )

        return perplexity["perplexity"]

    def compare_models(
            self,
            other_model: PreTrainedModel,
            dataset: Dict[str, Any],
            batch_size: int = 8
    ) -> Dict[str, Dict[str, float]]:
        """
        比较当前模型与另一个模型的性能

        Args:
            other_model: 要比较的另一个模型
            dataset: 评估数据集
            batch_size: 批次大小

        Returns:
            包含两个模型评估结果的字典
        """
        # 评估当前模型
        current_results = self.evaluate(dataset, batch_size)

        # 评估另一个模型
        other_evaluator = Web3ModelEvaluator(
            other_model, self.tokenizer, self.device, dataset, self.output_dir
        )
        other_results = other_evaluator.evaluate(dataset, batch_size)

        # 比较结果
        comparison = {
            "current_model": current_results,
            "other_model": other_results,
        }

        # 绘制比较图表
        self._plot_model_comparison(comparison)

        return comparison

    def _save_results(self, results: Dict[str, float]) -> None:
        """保存评估结果到文件"""
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Evaluation results saved to {results_path}")

    def _save_generation_samples(
            self,
            prompts: List[str],
            predictions: List[str],
            targets: List[str],
            num_samples: int = 20
    ) -> None:
        """保存生成样本用于人工检查"""
        samples_path = os.path.join(self.output_dir, "generation_samples.json")

        # 限制样本数量
        if len(prompts) > num_samples:
            indices = np.random.choice(len(prompts), num_samples, replace=False)
            samples = [
                {
                    "prompt": prompts[i],
                    "prediction": predictions[i],
                    "target": targets[i]
                }
                for i in indices
            ]
        else:
            samples = [
                {
                    "prompt": prompt,
                    "prediction": prediction,
                    "target": target
                }
                for prompt, prediction, target in zip(prompts, predictions, targets)
            ]

        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=4, ensure_ascii=False)

        print(f"Generation samples saved to {samples_path}")

    def _plot_confusion_matrix(self, labels: List[int], predictions: List[int]) -> None:
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 绘制图表
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")

        # 保存图表
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        print(f"Confusion matrix saved to {cm_path}")

    def _plot_model_comparison(self, comparison: Dict[str, Dict[str, float]]) -> None:
        """绘制模型比较图表"""
        metrics = list(comparison["current_model"].keys())
        current_values = [comparison["current_model"][metric] for metric in metrics]
        other_values = [comparison["other_model"][metric] for metric in metrics]

        # 绘制图表
        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, current_values, width, label="Current Model")
        plt.bar(x + width / 2, other_values, width, label="Other Model")

        plt.ylabel("Scores")
        plt.title("Model Comparison")
        plt.xticks(x, metrics, rotation=45)
        plt.legend()

        plt.tight_layout()

        # 保存图表
        comparison_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(comparison_path)
        plt.close()

        print(f"Model comparison saved to {comparison_path}")