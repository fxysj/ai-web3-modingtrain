import argparse
from data.dataset_loader import Web3DatasetLoader
from data.data_stream import Web3DataStream
from models.model_trainer import Web3ModelTrainer
from models.model_fine_tuner import Web3ModelFineTuner
from models.model_inference import Web3ModelInference
from config.base_config import BaseConfig
from config.finetune_config import FinetuneConfig
from config.inference_config import InferenceConfig
import asyncio


def train_base_model():
    """训练基础模型"""
    config = BaseConfig()
    dataset_loader = Web3DatasetLoader(config.tokenizer_name, config.max_length)

    # 加载数据集
    print("Loading base dataset...")
    dataset = dataset_loader.load_base_dataset()

    # 训练模型
    print("Training base model...")
    trainer = Web3ModelTrainer(config)
    model = trainer.train_base_model(dataset["train"], dataset["validation"])

    print(f"Base model trained and saved to {config.output_dir}")
    return model


def finetune_model():
    """微调模型"""
    config = FinetuneConfig()
    dataset_loader = Web3DatasetLoader(config.tokenizer_name, config.max_length)

    # 加载数据集
    print("Loading trading dataset for fine-tuning...")
    dataset = dataset_loader.load_trading_dataset()

    # 微调模型
    print("Fine-tuning model...")
    fine_tuner = Web3ModelFineTuner(config)
    model = fine_tuner.fine_tune_with_trading_data(dataset["train"], dataset["validation"])

    print(f"Fine-tuned model saved to {config.output_dir}")
    return model


async def continuous_finetuning():
    """基于实时数据流的持续微调"""
    config = FinetuneConfig()

    # 创建数据流
    data_stream = Web3DataStream(
        endpoint=config.stream_endpoint,
        api_key=config.api_key
    )

    # 持续微调
    print("Starting continuous fine-tuning...")
    fine_tuner = Web3ModelFineTuner(config)
    await fine_tuner.continuous_finetuning(data_stream)


def run_inference():
    """运行推理服务"""
    config = InferenceConfig()
    print(f"Starting inference service with model: {config.model_path}")

    # 这里可以启动API服务或直接进行推理
    # 简化示例，只进行一次推理测试
    inference = Web3ModelInference(config)

    # 示例提示
    prompt = "分析以太坊链上最近一周的交易趋势，特别是DeFi协议的表现。"
    result = inference.predict(prompt)

    print("Prompt:", prompt)
    print("Response:", result["response"])
    print(f"Generated {result['tokens_generated']} tokens")

    return result


def main():
    parser = argparse.ArgumentParser(description="Web3.0 AI Model Pipeline")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "finetune", "inference", "continuous_finetune"],
                        help="运行模式：训练基础模型、微调模型、运行推理或持续微调")

    args = parser.parse_args()

    if args.mode == "train":
        train_base_model()
    elif args.mode == "finetune":
        finetune_model()
    elif args.mode == "inference":
        run_inference()
    elif args.mode == "continuous_finetune":
        asyncio.run(continuous_finetuning())
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()