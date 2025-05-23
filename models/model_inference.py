# 模型推理
from vllm import LLM, SamplingParams
import torch
from fastapi import HTTPException


class Web3ModelInference:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.initialize_model()

    def initialize_model(self):
        """初始化推理模型"""
        try:
            self.llm = LLM(
                model=self.config.model_path,
                tokenizer=self.config.tokenizer_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype
            )

            # 设置默认采样参数
            self.default_sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty
            )

            print(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    def predict(self, prompt, sampling_params=None):
        """对给定的提示进行预测"""
        if self.llm is None:
            self.initialize_model()

        # 使用默认采样参数（如果未提供）
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        try:
            # 生成回答
            outputs = self.llm.generate([prompt], sampling_params)

            # 提取生成的文本
            generated_text = outputs[0].outputs[0].text
            return {
                "prompt": prompt,
                "response": generated_text,
                "tokens_generated": outputs[0].outputs[0].token_ids.shape[0]
            }
        except Exception as e:
            print(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    def batch_predict(self, prompts, sampling_params=None):
        """批量预测"""
        if self.llm is None:
            self.initialize_model()

        # 使用默认采样参数（如果未提供）
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        try:
            # 批量生成回答
            outputs = self.llm.generate(prompts, sampling_params)

            # 处理结果
            results = []
            for i, output in enumerate(outputs):
                results.append({
                    "prompt": prompts[i],
                    "response": output.outputs[0].text,
                    "tokens_generated": output.outputs[0].token_ids.shape[0]
                })

            return results
        except Exception as e:
            print(f"Error during batch inference: {e}")
            raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
