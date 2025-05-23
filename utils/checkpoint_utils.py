# 检查点工具
# tools/checkpoint_utils.py
import os
import argparse
import json
import torch
import shutil
import tempfile
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np
from huggingface_hub import snapshot_download, HfApi
from utils.logging_utils import setup_logger

# 设置日志
logger = setup_logger("checkpoint_utils")


class CheckpointManager:
    def __init__(self, model_path=None, tokenizer_path=None, output_dir=None):
        """初始化检查点管理器"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self, model_path=None, tokenizer_path=None, load_in_8bit=False, load_in_4bit=False):
        """加载模型和分词器"""
        model_path = model_path or self.model_path
        tokenizer_path = tokenizer_path or self.tokenizer_path

        if not model_path:
            raise ValueError("Model path is required")

        logger.info(f"Loading model from {model_path}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

        # 处理特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        torch_dtype = torch.float16

        if load_in_8bit:
            logger.info("Loading model in 8-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        elif load_in_4bit:
            logger.info("Loading model in 4-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            logger.info("Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )

        return self.model, self.tokenizer

    def merge_lora_checkpoint(self, base_model_path, lora_model_path, output_path=None):
        """合并LoRA检查点到基础模型"""
        output_path = output_path or self.output_dir or "./merged_model"

        logger.info(f"Merging LoRA checkpoint from {lora_model_path} to base model {base_model_path}")
        logger.info(f"Output will be saved to {output_path}")

        # 加载基础模型
        base_model, tokenizer = self.load_model(
            base_model_path,
            base_model_path,
            load_in_8bit=False,
            load_in_4bit=False
        )

        # 加载LoRA模型
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            device_map="auto"
        )

        # 合并LoRA权重
        logger.info("Merging LoRA weights into base model...")
        merged_model = lora_model.merge_and_unload()

        # 保存合并后的模型
        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("LoRA checkpoint merged successfully!")
        return output_path

    def convert_to_quantized_model(self, model_path, output_path=None, bits=4, group_size=32):
        """将模型转换为量化模型"""
        output_path = output_path or self.output_dir or f"./model_quantized_{bits}bit"

        logger.info(f"Converting model {model_path} to {bits}-bit quantization")
        logger.info(f"Output will be saved to {output_path}")

        # 加载模型
        model, tokenizer = self.load_model(
            model_path,
            model_path,
            load_in_8bit=(bits == 8),
            load_in_4bit=(bits == 4)
        )

        # 保存量化模型
        logger.info(f"Saving {bits}-bit quantized model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Model converted to {bits}-bit quantization successfully!")
        return output_path

    def compare_checkpoints(self, checkpoint_path1, checkpoint_path2, output_path=None):
        """比较两个检查点的差异"""
        output_path = output_path or self.output_dir or "./checkpoint_comparison"
        os.makedirs(output_path, exist_ok=True)

        logger.info(f"Comparing checkpoints: {checkpoint_path1} vs {checkpoint_path2}")

        # 加载两个模型
        model1, _ = self.load_model(checkpoint_path1)
        model2, _ = self.load_model(checkpoint_path2)

        # 获取模型状态字典
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        # 比较权重差异
        differences = {}
        total_diff = 0.0
        max_diff = 0.0
        diff_count = 0

        logger.info("Calculating weight differences...")
        for name, param1 in tqdm(state_dict1.items()):
            if name in state_dict2:
                param2 = state_dict2[name]

                # 确保参数形状相同
                if param1.shape == param2.shape:
                    # 计算差异
                    diff = torch.abs(param1 - param2)
                    mean_diff = torch.mean(diff).item()
                    max_param_diff = torch.max(diff).item()

                    differences[name] = {
                        "mean_diff": mean_diff,
                        "max_diff": max_param_diff,
                        "shape": list(param1.shape),
                        "dtype": str(param1.dtype)
                    }

                    total_diff += mean_diff
                    if max_param_diff > max_diff:
                        max_diff = max_param_diff
                    diff_count += 1
                else:
                    differences[name] = {
                        "error": f"Shape mismatch: {param1.shape} vs {param2.shape}"
                    }
            else:
                differences[name] = {
                    "error": "Parameter not found in second checkpoint"
                }

        # 计算总体差异统计
        avg_diff = total_diff / diff_count if diff_count > 0 else 0

        # 保存比较结果
        results = {
            "checkpoint1": checkpoint_path1,
            "checkpoint2": checkpoint_path2,
            "comparison_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {
                "total_parameters": len(state_dict1),
                "compared_parameters": diff_count,
                "average_difference": avg_diff,
                "max_difference": max_diff
            },
            "differences": differences
        }

        results_path = os.path.join(output_path, "comparison_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        # 生成摘要报告
        report_path = os.path.join(output_path, "comparison_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Checkpoint Comparison Report\n")
            f.write(f"============================\n\n")
            f.write(f"Checkpoint 1: {checkpoint_path1}\n")
            f.write(f"Checkpoint 2: {checkpoint_path2}\n")
            f.write(f"Comparison Time: {results['comparison_time']}\n\n")
            f.write(f"Statistics:\n")
            f.write(f"  Total Parameters: {results['stats']['total_parameters']}\n")
            f.write(f"  Compared Parameters: {results['stats']['compared_parameters']}\n")
            f.write(f"  Average Weight Difference: {results['stats']['average_difference']:.8f}\n")
            f.write(f"  Maximum Weight Difference: {results['stats']['max_difference']:.8f}\n\n")
            f.write(f"Top 10 Parameters with Largest Differences:\n")

            # 获取差异最大的10个参数
            sorted_params = sorted(
                differences.items(),
                key=lambda x: x[1].get("max_diff", 0),
                reverse=True
            )[:10]

            for name, info in sorted_params:
                if "max_diff" in info:
                    f.write(f"  - {name}: max_diff={info['max_diff']:.8f}, mean_diff={info['mean_diff']:.8f}\n")

        logger.info(f"Checkpoint comparison completed. Results saved to {results_path}")
        logger.info(f"Summary report saved to {report_path}")

        return results

    def validate_checkpoint(self, checkpoint_path, test_prompts=None):
        """验证检查点是否可以正常加载和生成"""
        logger.info(f"Validating checkpoint: {checkpoint_path}")

        try:
            # 尝试加载模型
            model, tokenizer = self.load_model(checkpoint_path)

            # 如果提供了测试提示，尝试生成响应
            if test_prompts:
                logger.info("Testing model generation...")
                model.eval()

                for i, prompt in enumerate(test_prompts):
                    logger.info(f"Test prompt {i + 1}: {prompt[:50]}...")

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=min(1024, inputs.input_ids.shape[1] + 512),
                            temperature=0.7,
                            top_p=0.9
                        )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Generated response: {response[:100]}...")

            logger.info(f"Checkpoint {checkpoint_path} is valid!")
            return True
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

    def download_checkpoint(self, model_id, output_dir=None, revision=None):
        """从Hugging Face Hub下载检查点"""
        output_dir = output_dir or self.output_dir or f"./hf_models/{model_id}"

        logger.info(f"Downloading checkpoint {model_id} to {output_dir}")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=output_dir,
                revision=revision,
                cache_dir="./hf_cache",
                local_dir_use_symlinks=False
            )

            logger.info(f"Checkpoint {model_id} downloaded successfully!")
            return output_dir
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return None

    def upload_checkpoint(self, checkpoint_path, repo_id, commit_message="Upload model checkpoint"):
        """上传检查点到Hugging Face Hub"""
        logger.info(f"Uploading checkpoint {checkpoint_path} to {repo_id}")

        try:
            api = HfApi()

            # 创建仓库（如果不存在）
            try:
                api.create_repo(repo_id, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create repo: {e}")

            # 上传文件
            api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=repo_id,
                commit_message=commit_message
            )

            logger.info(f"Checkpoint uploaded successfully to {repo_id}!")
            return True
        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Model Checkpoint Utility Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 合并LoRA检查点命令
    merge_parser = subparsers.add_parser("merge_lora", help="Merge LoRA checkpoint into base model")
    merge_parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    merge_parser.add_argument("--lora_model", type=str, required=True, help="LoRA model path")
    merge_parser.add_argument("--output_path", type=str, help="Output path for merged model")

    # 量化模型命令
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model to lower precision")
    quantize_parser.add_argument("--model_path", type=str, required=True, help="Model path")
    quantize_parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    quantize_parser.add_argument("--output_path", type=str, help="Output path for quantized model")

    # 比较检查点命令
    compare_parser = subparsers.add_parser("compare", help="Compare two model checkpoints")
    compare_parser.add_argument("--checkpoint1", type=str, required=True, help="First checkpoint path")
    compare_parser.add_argument("--checkpoint2", type=str, required=True, help="Second checkpoint path")
    compare_parser.add_argument("--output_path", type=str, help="Output path for comparison results")

    # 验证检查点命令
    validate_parser = subparsers.add_parser("validate", help="Validate model checkpoint")
    validate_parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint path")
    validate_parser.add_argument("--test_prompt", type=str, action="append", help="Test prompt(s)")

    # 下载检查点命令
    download_parser = subparsers.add_parser("download", help="Download model from Hugging Face Hub")
    download_parser.add_argument("--model_id", type=str, required=True, help="Model ID on Hugging Face Hub")
    download_parser.add_argument("--output_path", type=str, help="Local output path")
    download_parser.add_argument("--revision", type=str, help="Model revision")

    # 上传检查点命令
    upload_parser = subparsers.add_parser("upload", help="Upload model to Hugging Face Hub")
    upload_parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint path")
    upload_parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID")
    upload_parser.add_argument("--commit_message", type=str, default="Upload model checkpoint", help="Commit message")

    args = parser.parse_args()

    # 创建检查点管理器
    manager = CheckpointManager(output_dir="./checkpoint_utils_output")

    # 执行命令
    if args.command == "merge_lora":
        manager.merge_lora_checkpoint(args.base_model, args.lora_model, args.output_path)

    elif args.command == "quantize":
        manager.convert_to_quantized_model(args.model_path, args.output_path, args.bits)

    elif args.command == "compare":
        manager.compare_checkpoints(args.checkpoint1, args.checkpoint2, args.output_path)

    elif args.command == "validate":
        test_prompts = args.test_prompt or [
            "分析以太坊网络最近一个月的交易趋势",
            "解释Web3.0身份认证的主要技术挑战",
            "预测比特币价格在未来三个月的走势"
        ]
        manager.validate_checkpoint(args.checkpoint_path, test_prompts)

    elif args.command == "download":
        manager.download_checkpoint(args.model_id, args.output_path, args.revision)

    elif args.command == "upload":
        manager.upload_checkpoint(args.checkpoint_path, args.repo_id, args.commit_message)

    else:
        logger.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()