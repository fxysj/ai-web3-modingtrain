# 推理配置
# utils/inference_config.py
import os
import json
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    QuestionAnsweringPipeline,
    TextGenerationPipeline,
    pipeline
)
from .data_preprocessing import Web3DataPreprocessor


@dataclass
class InferenceArguments:
    """
    推理相关参数配置
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练或微调后的模型名称或路径"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "分词器名称，用于加载分词器"}
    )
    task_type: str = field(
        default="text-classification",
        metadata={"help": "任务类型 (text-classification, token-classification, question-answering, text-generation)"}
    )
    num_labels: Optional[int] = field(
        default=None, metadata={"help": "分类任务的标签数量"}
    )
    device: int = field(
        default=-1, metadata={"help": "使用的设备 (-1表示CPU, 0表示GPU 0, 1表示GPU 1, 依此类推)"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "推理批次大小"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "最大序列长度"}
    )
    padding: str = field(
        default="max_length", metadata={"help": "填充策略 (max_length, longest)"}
    )
    truncation: bool = field(
        default=True, metadata={"help": "是否截断序列"}
    )
    use_fp16: bool = field(
        default=False, metadata={"help": "是否使用半精度浮点数 (FP16) 推理"}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "是否使用Brain Floating Point (BF16) 推理"}
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "是否使用8位量化加载模型"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "是否使用4位量化加载模型"}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "文本生成温度参数"}
    )
    top_k: int = field(
        default=50, metadata={"help": "文本生成top-k参数"}
    )
    top_p: float = field(
        default=0.95, metadata={"help": "文本生成top-p参数"}
    )
    max_new_tokens: int = field(
        default=100, metadata={"help": "文本生成最大新生成token数"}
    )
    web3_preprocessing: bool = field(
        default=True, metadata={"help": "是否使用Web3.0特定预处理"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "缓存目录"}
    )


class Web3InferenceEngine:
    """
    Web3.0领域推理引擎
    """

    def __init__(self, args: InferenceArguments):
        """
        初始化推理引擎

        Args:
            args: 推理参数配置
        """
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.pipeline = self._create_pipeline()

        # 初始化Web3.0预处理器（如果需要）
        if args.web3_preprocessing:
            self.preprocessor = Web3DataPreprocessor(
                tokenizer=self.tokenizer,
                lower_case=True,
                remove_stopwords=True,
                normalize_web3_terms=True
            )
        else:
            self.preprocessor = None

    def _load_tokenizer(self) -> AutoTokenizer:
        """
        加载分词器

        Returns:
            分词器实例
        """
        tokenizer_name = self.args.tokenizer_name or self.args.model_name_or_path
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.args.cache_dir,
            use_fast=True
        )

    def _load_model(self) -> torch.nn.Module:
        """
        加载模型

        Returns:
            模型实例
        """
        # 设置加载参数
        load_kwargs = {}
        if self.args.use_fp16:
            load_kwargs["torch_dtype"] = torch.float16
        if self.args.use_bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16
        if self.args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        if self.args.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        # 根据任务类型加载相应模型
        if self.args.task_type == "text-classification":
            model_cls = AutoModelForSequenceClassification
            kwargs = {"num_labels": self.args.num_labels} if self.args.num_labels else {}
        elif self.args.task_type == "token-classification":
            model_cls = AutoModelForTokenClassification
            kwargs = {}
        elif self.args.task_type == "question-answering":
            model_cls = AutoModelForQuestionAnswering
            kwargs = {}
        elif self.args.task_type == "text-generation":
            model_cls = AutoModelForCausalLM
            kwargs = {}
        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")

        # 加载模型
        model = model_cls.from_pretrained(
            self.args.model_name_or_path,
            cache_dir=self.args.cache_dir,
            **load_kwargs,
            **kwargs
        )

        # 将模型移至指定设备
        model = model.to(self.device)
        model.eval()

        return model

    def _create_pipeline(self) -> Union[
        TextClassificationPipeline,
        TokenClassificationPipeline,
        QuestionAnsweringPipeline,
        TextGenerationPipeline
    ]:
        """
        创建推理pipeline

        Returns:
            推理pipeline实例
        """
        # 根据任务类型创建pipeline
        if self.args.task_type == "text-classification":
            return TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.args.device,
                return_all_scores=True
            )
        elif self.args.task_type == "token-classification":
            return TokenClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.args.device,
                aggregation_strategy="simple"
            )
        elif self.args.task_type == "question-answering":
            return QuestionAnsweringPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.args.device
            )
        elif self.args.task_type == "text-generation":
            return TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.args.device,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_new_tokens
            )
        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")

    def preprocess(self, inputs: List[str]) -> List[str]:
        """
        预处理输入文本

        Args:
            inputs: 输入文本列表

        Returns:
            预处理后的文本列表
        """
        if self.preprocessor is None:
            return inputs

        # 使用Web3.0预处理器处理文本
        return [self.preprocessor.clean_text(text) for text in inputs]

    def predict(self, inputs: List[str], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        执行推理

        Args:
            inputs: 输入文本列表
            batch_size: 批处理大小，默认为配置中的batch_size

        Returns:
            推理结果列表
        """
        # 使用指定的批处理大小，否则使用配置中的批处理大小
        batch_size = batch_size or self.args.batch_size

        # 预处理输入
        processed_inputs = self.preprocess(inputs)

        # 分批执行推理
        results = []
        start_time = time.time()

        for i in range(0, len(processed_inputs), batch_size):
            batch = processed_inputs[i:i + batch_size]

            # 根据任务类型执行推理
            if self.args.task_type == "text-classification":
                batch_results = self.pipeline(batch)
                # 处理分类结果，转换为更友好的格式
                formatted_results = []
                for result in batch_results:
                    labels = {item["label"]: item["score"] for item in result}
                    formatted_results.append({
                        "labels": labels,
                        "predicted_class": max(labels, key=labels.get),
                        "confidence": labels[max(labels, key=labels.get)]
                    })
                results.extend(formatted_results)

            elif self.args.task_type == "token-classification":
                # 直接使用NER结果
                results.extend(self.pipeline(batch))

            elif self.args.task_type == "question-answering":
                # 假设输入是问题和上下文的元组列表
                if all(isinstance(item, tuple) and len(item) == 2 for item in batch):
                    batch_results = []
                    for question, context in batch:
                        result = self.pipeline(question=question, context=context)
                        batch_results.append(result)
                    results.extend(batch_results)
                else:
                    raise ValueError("For question-answering task, inputs should be tuples of (question, context)")

            elif self.args.task_type == "text-generation":
                # 直接使用生成结果
                results.extend(self.pipeline(batch))

        # 计算推理时间
        inference_time = time.time() - start_time
        print(f"推理完成 - 样本数: {len(inputs)}, 总时间: {inference_time:.2f}秒, "
              f"平均时间: {inference_time / len(inputs):.4f}秒/样本")

        return results

    def postprocess(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        后处理推理结果

        Args:
            results: 推理结果列表

        Returns:
            后处理后的结果列表
        """
        # 针对Web3.0领域的结果后处理
        if self.args.task_type == "text-classification":
            # 示例：为Web3.0相关分类添加特殊处理
            for result in results:
                if "web3" in result["predicted_class"].lower():
                    # 增强Web3.0相关分类的置信度
                    result["confidence"] = min(1.0, result["confidence"] * 1.1)

        elif self.args.task_type == "token-classification":
            # 示例：处理Web3.0特定实体类型
            for result in results:
                for entity in result:
                    if entity["entity_group"] == "CRYPTO":
                        # 标准化加密货币名称
                        entity["word"] = entity["word"].upper()

        return results

    def run_inference(self, inputs: List[str], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        执行完整的推理流程（预处理、推理、后处理）

        Args:
            inputs: 输入文本列表
            batch_size: 批处理大小，默认为配置中的batch_size

        Returns:
            最终推理结果列表
        """
        # 执行预处理
        processed_inputs = self.preprocess(inputs)

        # 执行推理
        predictions = self.predict(processed_inputs, batch_size)

        # 执行后处理
        postprocessed_results = self.postprocess(predictions)

        return postprocessed_results


def get_default_web3_inference_config() -> InferenceArguments:
    """
    获取Web3.0领域的默认推理配置

    Returns:
        推理参数配置
    """
    return InferenceArguments(
        model_name_or_path="web3_fine_tuned_model",
        task_type="text-classification",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=8,
        max_seq_length=512,
        use_fp16=torch.cuda.is_available(),
        web3_preprocessing=True
    )


# 示例使用
if __name__ == "__main__":
    # 获取默认Web3.0推理配置
    inference_args = get_default_web3_inference_config()

    # 创建推理引擎
    inference_engine = Web3InferenceEngine(inference_args)

    # 示例输入
    example_inputs = [
        "以太坊Layer 2解决方案将如何影响Gas费？",
        "比特币价格突破6万美元，是否应该买入？",
        "这是一个关于DAO治理的讨论，欢迎参与！"
    ]

    # 执行推理
    results = inference_engine.run_inference(example_inputs)

    # 打印结果
    for i, result in enumerate(results):
        print(f"输入: {example_inputs[i]}")
        print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 50)

    # 示例：问答任务
    if inference_args.task_type == "question-answering":
        qa_inputs = [
            ("什么是DeFi？", "DeFi是去中心化金融的缩写，是基于区块链的金融系统。"),
            ("以太坊和比特币有什么区别？", "以太坊支持智能合约，而比特币主要作为数字货币。")
        ]

        qa_results = inference_engine.run_inference(qa_inputs)

        for i, result in enumerate(qa_results):
            print(f"问题: {qa_inputs[i][0]}")
            print(f"上下文: {qa_inputs[i][1][:50]}...")
            print(f"答案: {result['answer']}")
            print("-" * 50)