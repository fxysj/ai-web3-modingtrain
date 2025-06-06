# 评估指标工具
# utils/metrics_utils.py
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_recall_curve
)
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
from datasets import load_metric
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 确保nltk数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextGenerationMetrics:
    """文本生成任务的评估指标"""

    @staticmethod
    def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE指标（ROUGE-1、ROUGE-2、ROUGE-L）"""
        if not predictions or not references:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)

        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }

    @staticmethod
    def calculate_bleu(predictions: List[str], references: List[str]) -> float:
        """计算BLEU指标"""
        if not predictions or not references:
            return 0.0

        # 将文本转换为token列表
        pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]

        # 使用平滑函数处理短文本
        sf = SmoothingFunction().method4
        return corpus_bleu(ref_tokens, pred_tokens, smoothing_function=sf)

    @staticmethod
    def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BERTScore（需要安装bert-score库）"""
        try:
            from bert_score import score

            if not predictions or not references:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            P, R, F1 = score(predictions, references, lang="en", verbose=False)
            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
        except ImportError:
            print("Warning: bert-score library not found. Skipping BERTScore calculation.")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    @staticmethod
    def calculate_perplexity(texts: List[str], model, tokenizer) -> float:
        """计算文本的困惑度"""
        if not texts:
            return float('inf')

        device = next(model.parameters()).device
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            total_loss += loss.item() * inputs.input_ids.size(1)
            total_tokens += inputs.input_ids.size(1)

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    @staticmethod
    def calculate_diversity(predictions: List[str]) -> Dict[str, float]:
        """计算文本多样性指标（唯一unigram和bigram的比例）"""
        if not predictions:
            return {"unique_unigrams": 0.0, "unique_bigrams": 0.0}

        all_unigrams = set()
        all_bigrams = set()
        total_unigrams = 0
        total_bigrams = 0

        for pred in predictions:
            tokens = nltk.word_tokenize(pred.lower())
            unigrams = tokens
            bigrams = list(nltk.bigrams(tokens))

            all_unigrams.update(unigrams)
            all_bigrams.update(bigrams)
            total_unigrams += len(unigrams)
            total_bigrams += len(bigrams)

        return {
            "unique_unigrams": len(all_unigrams) / total_unigrams if total_unigrams > 0 else 0.0,
            "unique_bigrams": len(all_bigrams) / total_bigrams if total_bigrams > 0 else 0.0
        }


class ClassificationMetrics:
    """分类任务的评估指标"""

    @staticmethod
    def calculate_accuracy(y_true: List[Union[int, str]], y_pred: List[Union[int, str]]) -> float:
        """计算准确率"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_f1(y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
                     average: str = 'weighted') -> float:
        """计算F1分数"""
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_precision(y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
                            average: str = 'weighted') -> float:
        """计算精确率"""
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_recall(y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
                         average: str = 'weighted') -> float:
        """计算召回率"""
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def get_classification_report(y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
                                  labels: Optional[List[Union[int, str]]] = None) -> str:
        """获取分类报告"""
        return classification_report(y_true, y_pred, labels=labels)

    @staticmethod
    def plot_confusion_matrix(y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
                              labels: Optional[List[Union[int, str]]] = None,
                              output_path: Optional[str] = None) -> None:
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels if labels else sorted(set(y_true)),
                    yticklabels=labels if labels else sorted(set(y_true)))
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @staticmethod
    def calculate_auc_roc(y_true: List[Union[int, float]], y_score: List[float],
                          multi_class: str = 'ovr') -> float:
        """计算ROC曲线下面积（AUC-ROC）"""
        return roc_auc_score(y_true, y_score, multi_class=multi_class)

    @staticmethod
    def plot_precision_recall_curve(y_true: List[Union[int, float]], y_score: List[float],
                                    output_path: Optional[str] = None) -> None:
        """绘制精确率-召回率曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, marker='.', label='Model')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


class RegressionMetrics:
    """回归任务的评估指标"""

    @staticmethod
    def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
        """计算均方误差(MSE)"""
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
        """计算均方根误差(RMSE)"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
        """计算平均绝对误差(MAE)"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_r2(y_true: List[float], y_pred: List[float]) -> float:
        """计算R²分数"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_mape(y_true: List[float], y_pred: List[float]) -> float:
        """计算平均绝对百分比误差(MAPE)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def calculate_smape(y_true: List[float], y_pred: List[float]) -> float:
        """计算对称平均绝对百分比误差(SMAPE)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    @staticmethod
    def plot_prediction_error(y_true: List[float], y_pred: List[float],
                              output_path: Optional[str] = None) -> None:
        """绘制预测误差图"""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')

        plt.subplot(1, 2, 2)
        errors = y_pred - y_true
        plt.hist(errors, bins=20)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


class Web3Metrics:
    """Web3.0特定领域的评估指标"""

    @staticmethod
    def calculate_prediction_relevance(predictions: List[str], references: List[str],
                                       domain_terms: List[str]) -> Dict[str, float]:
        """计算预测与Web3.0领域术语的相关性"""
        if not predictions or not references:
            return {"relevance_score": 0.0, "coverage_ratio": 0.0}

        total_relevant_terms = 0
        total_domain_terms = len(domain_terms)
        domain_term_set = set(term.lower() for term in domain_terms)

        for pred in predictions:
            pred_terms = set(nltk.word_tokenize(pred.lower()))
            relevant_terms = pred_terms.intersection(domain_term_set)
            total_relevant_terms += len(relevant_terms)

        # 计算平均每个预测中涉及的领域术语数量
        relevance_score = total_relevant_terms / len(predictions) if predictions else 0.0

        # 计算领域术语覆盖率（至少在一个预测中出现的术语比例）
        covered_terms = set()
        for pred in predictions:
            pred_terms = set(nltk.word_tokenize(pred.lower()))
            covered_terms.update(pred_terms.intersection(domain_term_set))

        coverage_ratio = len(covered_terms) / total_domain_terms if total_domain_terms > 0 else 0.0

        return {
            "relevance_score": relevance_score,
            "coverage_ratio": coverage_ratio
        }

    @staticmethod
    def calculate_numeric_accuracy(predictions: List[float], references: List[float],
                                   tolerance: float = 0.05) -> float:
        """计算Web3.0数值预测的准确性（考虑一定的容差范围）"""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0

        correct_count = 0
        for pred, ref in zip(predictions, references):
            if ref == 0:
                # 处理参考值为0的情况
                is_correct = (pred == 0)
            else:
                # 计算相对误差
                relative_error = abs(pred - ref) / abs(ref)
                is_correct = (relative_error <= tolerance)

            if is_correct:
                correct_count += 1

        return correct_count / len(predictions)

    @staticmethod
    def evaluate_chain_analysis_accuracy(predictions: List[Dict[str, Any]],
                                         references: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估区块链分析的准确性（例如交易模式识别）"""
        if not predictions or not references or len(predictions) != len(references):
            return {"accuracy": 0.0, "f1": 0.0}

        # 提取关键指标
        y_true = []
        y_pred = []

        for pred, ref in zip(predictions, references):
            # 假设每个预测和参考都是包含多个指标的字典
            # 这里简化为二分类问题：预测是否正确识别了某种模式
            pred_pattern = pred.get("pattern_detected", False)
            ref_pattern = ref.get("pattern_detected", False)

            y_true.append(ref_pattern)
            y_pred.append(pred_pattern)

        # 计算分类指标
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')

        return {
            "accuracy": accuracy,
            "f1": f1
        }

    @staticmethod
    def evaluate_market_prediction(predictions: List[Dict[str, float]],
                                   references: List[Dict[str, float]]) -> Dict[str, float]:
        """评估市场预测的准确性（例如价格走势预测）"""
        if not predictions or not references or len(predictions) != len(references):
            return {"direction_accuracy": 0.0, "rmse": 0.0}

        direction_correct = 0
        price_errors = []

        for pred, ref in zip(predictions, references):
            # 评估价格变动方向
            pred_direction = pred.get("price_direction", 0)  # 1=上涨, 0=持平, -1=下跌
            ref_direction = ref.get("price_direction", 0)

            if pred_direction == ref_direction:
                direction_correct += 1

            # 评估价格预测误差
            pred_price = pred.get("price", 0)
            ref_price = ref.get("price", 0)

            if ref_price != 0:
                price_errors.append((pred_price - ref_price) ** 2)

        # 计算方向预测准确率
        direction_accuracy = direction_correct / len(predictions) if predictions else 0.0

        # 计算价格预测RMSE
        rmse = np.sqrt(np.mean(price_errors)) if price_errors else 0.0

        return {
            "direction_accuracy": direction_accuracy,
            "rmse": rmse
        }


class MetricsVisualizer:
    """评估指标可视化工具"""

    @staticmethod
    def plot_text_metrics(metrics: Dict[str, float], output_path: Optional[str] = None) -> None:
        """可视化文本生成指标"""
        if not metrics:
            return

        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.title('Text Generation Metrics')
        plt.ylim(0, 1)  # 大多数文本指标范围在0-1之间

        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @staticmethod
    def plot_comparison(metrics_list: List[Dict[str, float]], labels: List[str],
                        output_path: Optional[str] = None) -> None:
        """比较不同模型或配置的指标"""
        if not metrics_list or len(metrics_list) != len(labels):
            return

        # 确保所有字典有相同的键
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # 创建数据框
        data = {}
        for key in all_keys:
            data[key] = [metrics.get(key, 0) for metrics in metrics_list]

        df = pd.DataFrame(data, index=labels)

        # 绘制柱状图
        ax = df.plot(kind='bar', figsize=(12, 8))
        plt.xlabel('Models/Configs')
        plt.ylabel('Scores')
        plt.title('Metrics Comparison')
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')

        # 添加数值标签
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f"{height:.4f}", (x + width / 2, y + height), ha='center', va='bottom')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


# 整合所有评估功能的函数
def compute_metrics(task_type: str, predictions: List[Any], references: List[Any],
                    **kwargs) -> Dict[str, float]:
    """
    统一计算评估指标的函数

    Args:
        task_type: 任务类型，如"text_generation", "classification", "regression", "web3_analysis"
        predictions: 模型预测结果
        references: 参考标准答案
        **kwargs: 其他可能的参数，如domain_terms, tolerance等

    Returns:
        包含各种评估指标的字典
    """
    metrics = {}

    if task_type == "text_generation":
        metrics.update(TextGenerationMetrics.calculate_rouge(predictions, references))
        metrics["bleu"] = TextGenerationMetrics.calculate_bleu(predictions, references)
        metrics.update(TextGenerationMetrics.calculate_diversity(predictions))

        # 计算BERTScore（如果安装了库）
        bert_score = TextGenerationMetrics.calculate_bert_score(predictions, references)
        metrics.update({f"bert_{k}": v for k, v in bert_score.items()})

        # 如果提供了模型和分词器，计算困惑度
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model and tokenizer:
            metrics["perplexity"] = TextGenerationMetrics.calculate_perplexity(predictions, model, tokenizer)

        # 如果是Web3.0相关的文本生成，计算领域相关性
        domain_terms = kwargs.get("domain_terms")
        if domain_terms:
            web3_metrics = Web3Metrics.calculate_prediction_relevance(predictions, references, domain_terms)
            metrics.update(web3_metrics)

    elif task_type == "classification":
        metrics["accuracy"] = ClassificationMetrics.calculate_accuracy(predictions, references)
        metrics["f1"] = ClassificationMetrics.calculate_f1(predictions, references)
        metrics["precision"] = ClassificationMetrics.calculate_precision(predictions, references)
        metrics["recall"] = ClassificationMetrics.calculate_recall(predictions, references)

        # 如果是二分类问题，计算AUC-ROC
        if len(set(references)) == 2:
            # 假设predictions是概率分数
            if all(isinstance(p, float) for p in predictions):
                metrics["auc_roc"] = ClassificationMetrics.calculate_auc_roc(references, predictions)

    elif task_type == "regression":
        metrics["mse"] = RegressionMetrics.calculate_mse(predictions, references)
        metrics["rmse"] = RegressionMetrics.calculate_rmse(predictions, references)
        metrics["mae"] = RegressionMetrics.calculate_mae(predictions, references)
        metrics["r2"] = RegressionMetrics.calculate_r2(predictions, references)

        # 计算百分比误差（确保没有零值）
        if all(ref != 0 for ref in references):
            metrics["mape"] = RegressionMetrics.calculate_mape(predictions, references)
            metrics["smape"] = RegressionMetrics.calculate_smape(predictions, references)

        # 如果是Web3.0数值预测，计算带容差的准确率
        tolerance = kwargs.get("tolerance", 0.05)
        metrics["numeric_accuracy"] = Web3Metrics.calculate_numeric_accuracy(predictions, references, tolerance)

    elif task_type == "web3_analysis":
        # Web3.0特定的分析任务
        analysis_type = kwargs.get("analysis_type", "general")

        if analysis_type == "chain_analysis":
            # 区块链分析（如交易模式识别）
            metrics.update(Web3Metrics.evaluate_chain_analysis_accuracy(predictions, references))
        elif analysis_type == "market_prediction":
            # 市场预测（如价格走势）
            metrics.update(Web3Metrics.evaluate_market_prediction(predictions, references))
        else:
            # 通用Web3.0分析
            if all(isinstance(p, str) for p in predictions) and all(isinstance(r, str) for r in references):
                # 文本分析
                metrics.update(TextGenerationMetrics.calculate_rouge(predictions, references))
                metrics["bleu"] = TextGenerationMetrics.calculate_bleu(predictions, references)

                domain_terms = kwargs.get("domain_terms")
                if domain_terms:
                    metrics.update(Web3Metrics.calculate_prediction_relevance(predictions, references, domain_terms))
            elif all(isinstance(p, dict) for p in predictions) and all(isinstance(r, dict) for r in references):
                # 结构化数据分析
                metrics.update(Web3Metrics.evaluate_chain_analysis_accuracy(predictions, references))

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return metrics