# 示例：如何使用模型评估器
from models.model_evaluator import Web3ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("your_model_path")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

# 加载评估数据集
dataset = load_dataset("0xscope/web3-trading-analysis", split="validation")

# 准备评估数据
eval_data = {
    "text": [item["question"] for item in dataset],
    "target": [item["answer"] for item in dataset],
}

# 初始化评估器
evaluator = Web3ModelEvaluator(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_data,
    output_dir="./web3_model_evaluation"
)

# 执行评估
results = evaluator.evaluate()

# 打印评估结果
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# 比较两个模型
# other_model = AutoModelForCausalLM.from_pretrained("another_model_path")
# comparison = evaluator.compare_models(other_model, eval_data)