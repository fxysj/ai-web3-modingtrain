# 运行推理
# scripts/run_inference.py
import argparse
import torch
import json
import time
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from utils.logging_utils import setup_logger
from utils.metrics_utils import calculate_rouge, calculate_bleu
from data.data_preprocessor import Web3DataPreprocessor

# 设置日志
logger = setup_logger("run_inference")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run inference with a Web3.0 model")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="./web3_finetuned_model",
                        help="微调后的模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径，默认为模型路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备 (cuda/cpu)")

    # 输入配置
    parser.add_argument("--input_file", type=str, default="web3_queries.jsonl",
                        help="输入查询文件 (JSONL格式，每行包含'query'字段)")
    parser.add_argument("--output_file", type=str, default="inference_results.jsonl",
                        help="结果保存文件")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批量推理批次大小")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度 (0.0-1.0，越低越确定)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="核采样概率")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成长度")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="束搜索数量 (用于确定性生成)")
    parser.add_argument("--use_vllm", action="store_true",
                        help="使用vLLM进行高性能推理")

    # 评估配置
    parser.add_argument("--reference_file", type=str,
                        help="参考标准答案文件 (用于计算ROUGE/BLEU)")
    parser.add_argument("--save_samples", type=int, default=20,
                        help="保存示例结果数量")

    return parser.parse_args()


def load_input_data(input_file):
    """加载输入查询数据"""
    logger.info(f"Loading input data from {input_file}")
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        raise ValueError("Input file is empty")

    # 确保包含'query'字段
    if "query" not in data[0]:
        raise ValueError("Input data must contain 'query' field")

    return Dataset.from_list(data)


def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    logger.info(f"Loading model from {args.model_path}")
    tokenizer_path = args.tokenizer_path or args.model_path

    if args.use_vllm:
        # 使用vLLM加载模型（适用于推理优化）
        llm = LLM(
            model=args.model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=torch.cuda.device_count() if args.device == "cuda" else 1,
            dtype="float16" if torch.cuda.is_available() else "float32",
            trust_remote_code=True
        )
        return None, tokenizer_path, llm
    else:
        # 使用Hugging Face原生加载（适用于需要自定义生成逻辑的场景）
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # 处理特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer, None


def generate_with_hf(model, tokenizer, query, generation_config):
    """使用Hugging Face原生生成"""
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_with_vllm(llm, tokenizer, query, sampling_params):
    """使用vLLM生成"""
    outputs = llm.generate([query], sampling_params)
    return outputs[0].outputs[0].text


def run_batch_inference(args, model, tokenizer, llm, dataset):
    """执行批量推理"""
    results = []
    total_queries = len(dataset)
    logger.info(f"Starting inference for {total_queries} queries")

    # 设置生成参数
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        num_beams=args.num_beams,
        eos_token_id=tokenizer.eos_token_id if tokenizer else None,
        pad_token_id=tokenizer.pad_token_id if tokenizer else None
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_beams=args.num_beams,
    )

    # 分批处理
    for i in tqdm(range(0, total_queries, args.batch_size)):
        batch = dataset[i:i + args.batch_size]
        queries = [item["query"] for item in batch]

        # 生成回答
        if args.use_vllm:
            if not llm:
                raise ValueError("vLLM model not loaded")
            responses = [generate_with_vllm(llm, tokenizer, q, sampling_params) for q in queries]
        else:
            if not model or not tokenizer:
                raise ValueError("Hugging Face model or tokenizer not loaded")
            responses = [generate_with_hf(model, tokenizer, q, generation_config) for q in queries]

        # 保存结果
        for query, response in zip(queries, responses):
            results.append({
                "query": query,
                "response": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": args.model_path,
                "parameters": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "num_beams": args.num_beams
                }
            })

    return results


def save_results(results, output_file, save_samples=20):
    """保存推理结果"""
    logger.info(f"Saving {len(results)} results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 保存示例结果
    if save_samples > 0:
        samples = results[:save_samples]
        sample_file = output_file.replace(".jsonl", "_samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {save_samples} sample results to {sample_file}")


def evaluate_results(results, reference_file):
    """评估推理结果"""
    if not reference_file:
        logger.info("No reference file provided, skipping evaluation")
        return

    logger.info(f"Loading reference data from {reference_file}")
    references = []
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            references.append(json.loads(line)["answer"])

    # 提取预测结果
    predictions = [item["response"] for item in results]

    # 计算ROUGE指标
    rouge_scores = calculate_rouge(predictions, references)
    logger.info(f"ROUGE Scores: {rouge_scores}")

    # 计算BLEU分数
    bleu_score = calculate_bleu(predictions, references)
    logger.info(f"BLEU Score: {bleu_score:.2f}")

    return {
        "rouge": rouge_scores,
        "bleu": bleu_score
    }


def main():
    """主函数"""
    args = parse_args()

    # 加载输入数据
    dataset = load_input_data(args.input_file)

    # 加载模型和分词器
    model, tokenizer, llm = load_model_and_tokenizer(args)

    # 执行推理
    results = run_batch_inference(args, model, tokenizer, llm, dataset)

    # 保存结果
    save_results(results, args.output_file, args.save_samples)

    # 评估结果
    if args.reference_file:
        eval_metrics = evaluate_results(results, args.reference_file)
        with open("evaluation_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=4)
        logger.info("Evaluation metrics saved to evaluation_metrics.json")

    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()