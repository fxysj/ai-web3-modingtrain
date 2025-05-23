# 数据预处理工具
# utils/data_preprocessing.py
import re
import json
import os
import html
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import emoji
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from transformers import PreTrainedTokenizer

# 确保nltk数据已下载
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class Web3DataPreprocessor:
    """Web3.0领域文本数据预处理工具"""

    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None,
                 language: str = 'english', lower_case: bool = True,
                 remove_stopwords: bool = True, normalize_web3_terms: bool = True):
        """
        初始化数据预处理器

        Args:
            tokenizer: 可选的分词器，用于编码文本
            language: 语言设置，影响停用词等
            lower_case: 是否将文本转换为小写
            remove_stopwords: 是否移除停用词
            normalize_web3_terms: 是否标准化Web3.0领域术语
        """
        self.tokenizer = tokenizer
        self.language = language
        self.lower_case = lower_case
        self.remove_stopwords = remove_stopwords
        self.normalize_web3_terms = normalize_web3_terms

        # 加载停用词
        self.stop_words = set(stopwords.words(language))

        # Web3.0领域术语映射表（用于标准化）
        self.web3_term_mapping = {
            "eth": "以太坊",
            "btc": "比特币",
            "defi": "去中心化金融",
            "nft": "非同质化代币",
            "dao": "去中心化自治组织",
            "dapp": "去中心化应用",
            "gas": "Gas费",
            "wallet": "钱包",
            "blockchain": "区块链",
            "smart contract": "智能合约",
            "mining": "挖矿",
            "validator": "验证者",
            "staking": "质押",
            "liquidity": "流动性",
            "oracle": "预言机",
            "bridge": "跨链桥",
            "layer 1": "第一层",
            "layer 2": "第二层",
            "protocol": "协议",
            "token": "代币",
            "transaction": "交易",
            "address": "地址",
            "私钥": "私钥",
            "公钥": "公钥",
            "hash": "哈希",
            "空投": "空投",
            "流动性挖矿": "流动性挖矿",
            "yield farming": "收益耕作",
            "impermanent loss": "无常损失"
        }

        # 正则表达式模式
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.eth_address_pattern = re.compile(r'0x[a-fA-F0-9]{40}')
        self.number_pattern = re.compile(r'\d+(\.\d+)?')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')

        # 表情符号映射（将表情符号转换为文本描述）
        self.emoji_mapping = {
            "🚀": "上升",
            "📉": "下降",
            "💎": "加密货币",
            "🐂": "牛市",
            "🐻": "熊市",
            "💰": "金钱",
            "💸": "支出",
            "💹": "图表上升",
            "📊": "图表",
            "🔒": "安全",
            "🔓": "不安全",
            "⚠️": "警告",
            "✅": "成功",
            "❌": "失败"
        }

    def clean_text(self, text: str) -> str:
        """
        执行基本文本清洗

        Args:
            text: 输入文本

        Returns:
            清洗后的文本
        """
        # 解码HTML实体
        text = html.unescape(text)

        # 移除HTML标签
        text = BeautifulSoup(text, "html.parser").get_text()

        # 移除URL
        text = self.url_pattern.sub('', text)

        # 移除电子邮件
        text = self.email_pattern.sub('', text)

        # 移除提及
        text = self.mention_pattern.sub('', text)

        # 移除ETH地址
        text = self.eth_address_pattern.sub('ETH_ADDRESS', text)

        # 转换表情符号为文本
        text = emoji.demojize(text)
        for emoji_symbol, text_desc in self.emoji_mapping.items():
            text = text.replace(emoji_symbol, f" {text_desc} ")

        # 标准化Web3.0术语
        if self.normalize_web3_terms:
            for term, replacement in self.web3_term_mapping.items():
                # 使用单词边界确保精确匹配
                term_pattern = re.compile(rf'\b{term}\b', re.IGNORECASE)
                text = term_pattern.sub(replacement, text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def normalize_text(self, text: str) -> str:
        """
        执行文本标准化

        Args:
            text: 输入文本

        Returns:
            标准化后的文本
        """
        # 转换为小写（如果需要）
        if self.lower_case:
            text = text.lower()

        # 移除标点符号
        text = self.punctuation_pattern.sub(' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        分词文本

        Args:
            text: 输入文本

        Returns:
            分词后的列表
        """
        # 使用NLTK分词器
        tokens = word_tokenize(text, language=self.language)

        # 移除停用词（如果需要）
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        return tokens

    def encode_text(self, text: str, max_length: int = 128,
                    padding: str = 'max_length', truncation: bool = True) -> Dict[str, List[int]]:
        """
        使用分词器编码文本

        Args:
            text: 输入文本
            max_length: 最大序列长度
            padding: 填充方式
            truncation: 是否截断

        Returns:
            编码后的输入字典
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for encoding text")

        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

    def extract_numeric_features(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取数值特征

        Args:
            text: 输入文本

        Returns:
            包含数值特征的字典
        """
        features = {}

        # 提取所有数字
        numbers = [float(num) for num in self.number_pattern.findall(text)]

        if numbers:
            features["num_count"] = len(numbers)
            features["num_sum"] = sum(numbers)
            features["num_mean"] = sum(numbers) / len(numbers)
            features["num_max"] = max(numbers)
            features["num_min"] = min(numbers)
        else:
            features["num_count"] = 0
            features["num_sum"] = 0.0
            features["num_mean"] = 0.0
            features["num_max"] = 0.0
            features["num_min"] = 0.0

        # 提取百分比
        percentages = [float(p[:-1]) for p in self.percentage_pattern.findall(text)]
        features["percentages"] = percentages

        return features

    def extract_web3_features(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取Web3.0特定特征

        Args:
            text: 输入文本

        Returns:
            包含Web3.0特征的字典
        """
        features = {}

        # 检查是否包含ETH地址
        features["has_eth_address"] = 1 if self.eth_address_pattern.search(text) else 0

        # 计算Web3.0术语出现次数
        term_counts = {}
        total_terms = 0

        for term in self.web3_term_mapping.values():
            count = text.lower().count(term.lower())
            term_counts[term] = count
            total_terms += count

        features["web3_term_counts"] = term_counts
        features["web3_term_total"] = total_terms

        # 检查是否包含协议名称
        protocols = ["以太坊", "比特币", "币安智能链", "Polygon", "Avalanche", "Solana", "Cosmos"]
        features["protocols_mentioned"] = [p for p in protocols if p.lower() in text.lower()]
        features["num_protocols"] = len(features["protocols_mentioned"])

        return features

    def process_text(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        处理单个文本，返回清洗、标准化、编码后的结果

        Args:
            text: 输入文本
            max_length: 最大序列长度

        Returns:
            处理后的特征字典
        """
        processed = {}

        # 清洗文本
        cleaned_text = self.clean_text(text)
        processed["cleaned_text"] = cleaned_text

        # 标准化文本
        normalized_text = self.normalize_text(cleaned_text)
        processed["normalized_text"] = normalized_text

        # 分词
        tokens = self.tokenize_text(normalized_text)
        processed["tokens"] = tokens

        # 提取数值特征
        numeric_features = self.extract_numeric_features(cleaned_text)
        processed.update({f"numeric_{k}": v for k, v in numeric_features.items()})

        # 提取Web3.0特征
        web3_features = self.extract_web3_features(cleaned_text)
        processed.update({f"web3_{k}": v for k, v in web3_features.items()})

        # 编码文本（如果有分词器）
        if self.tokenizer is not None:
            encoding = self.encode_text(cleaned_text, max_length=max_length)
            processed["input_ids"] = encoding["input_ids"].squeeze(0).tolist()
            processed["attention_mask"] = encoding["attention_mask"].squeeze(0).tolist()

        return processed

    def process_batch(self, texts: List[str], max_length: int = 128) -> List[Dict[str, Any]]:
        """
        批量处理文本

        Args:
            texts: 输入文本列表
            max_length: 最大序列长度

        Returns:
            处理后的特征字典列表
        """
        return [self.process_text(text, max_length=max_length) for text in texts]


class Web3DatasetProcessor:
    """Web3.0领域数据集处理器"""

    def __init__(self, preprocessor: Web3DataPreprocessor):
        """
        初始化数据集处理器

        Args:
            preprocessor: 数据预处理器
        """
        self.preprocessor = preprocessor

    def process_dataset(self, dataset: List[Dict[str, Any]], text_field: str = "text",
                        label_field: str = "label", max_length: int = 128) -> List[Dict[str, Any]]:
        """
        处理整个数据集

        Args:
            dataset: 数据集，每个元素是一个字典
            text_field: 文本字段名
            label_field: 标签字段名
            max_length: 最大序列长度

        Returns:
            处理后的数据集
        """
        processed_data = []

        for item in dataset:
            # 确保数据项包含文本字段
            if text_field not in item:
                continue

            # 处理文本
            processed_text = self.preprocessor.process_text(item[text_field], max_length=max_length)

            # 创建处理后的样本
            processed_item = {
                **processed_text,
                "original_text": item[text_field]
            }

            # 添加标签（如果存在）
            if label_field in item:
                processed_item["label"] = item[label_field]

            # 添加其他字段（如果有）
            for key, value in item.items():
                if key not in [text_field, label_field]:
                    processed_item[key] = value

            processed_data.append(processed_item)

        return processed_data

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text",
                          label_column: str = "label", max_length: int = 128) -> pd.DataFrame:
        """
        处理DataFrame格式的数据集

        Args:
            df: 输入DataFrame
            text_column: 文本列名
            label_column: 标签列名
            max_length: 最大序列长度

        Returns:
            处理后的DataFrame
        """
        # 处理每一行
        processed_rows = []

        for _, row in df.iterrows():
            # 处理文本
            processed_text = self.preprocessor.process_text(row[text_column], max_length=max_length)

            # 创建处理后的行
            processed_row = {
                **processed_text,
                "original_text": row[text_column]
            }

            # 添加标签（如果存在）
            if label_column in row:
                processed_row["label"] = row[label_column]

            # 添加其他列
            for col in df.columns:
                if col not in [text_column, label_column]:
                    processed_row[col] = row[col]

            processed_rows.append(processed_row)

        # 转换为DataFrame
        processed_df = pd.DataFrame(processed_rows)

        return processed_df

    def create_torch_dataset(self, dataset: List[Dict[str, Any]],
                             text_field: str = "cleaned_text",
                             label_field: str = "label") -> torch.utils.data.Dataset:
        """
        创建PyTorch数据集

        Args:
            dataset: 数据集，每个元素是一个字典
            text_field: 文本字段名
            label_field: 标签字段名

        Returns:
            PyTorch数据集
        """
        from torch.utils.data import Dataset

        class Web3Dataset(Dataset):
            def __init__(self, data, preprocessor, text_field, label_field):
                self.data = data
                self.preprocessor = preprocessor
                self.text_field = text_field
                self.label_field = label_field

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = item[self.text_field]

                # 编码文本
                encoding = self.preprocessor.encode_text(text)

                # 创建样本
                sample = {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0)
                }

                # 添加标签（如果存在）
                if self.label_field in item:
                    sample["labels"] = torch.tensor(item[self.label_field])

                return sample

        return Web3Dataset(dataset, self.preprocessor, text_field, label_field)


# 辅助函数
def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载数据

    Args:
        file_path: 文件路径

    Returns:
        加载的数据列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    将数据保存到JSON文件

    Args:
        data: 数据列表
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_dataset(dataset: List[Dict[str, Any]], train_ratio: float = 0.8,
                  val_ratio: float = 0.1, test_ratio: float = 0.1,
                  seed: int = 42) -> Tuple[List[Dict[str, Any]],
List[Dict[str, Any]],
List[Dict[str, Any]]]:
    """
    分割数据集为训练集、验证集和测试集

    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        分割后的训练集、验证集和测试集
    """
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    # 打乱数据集
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # 计算分割点
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)

    # 分割数据集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建分割后的数据集
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]

    return train_data, val_data, test_data


# 示例使用
if __name__ == "__main__":
    # 从transformers加载分词器
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 创建预处理器
    preprocessor = Web3DataPreprocessor(
        tokenizer=tokenizer,
        language='chinese',
        lower_case=True,
        remove_stopwords=True,
        normalize_web3_terms=True
    )

    # 示例文本
    sample_text = "📢【重要通知】以太坊Layer 2解决方案将降低Gas费并提高交易速度！" + \
                  "当前ETH价格为$1,800，较昨日上涨2.5%。0x1234...abcd是一个测试地址。"

    # 处理单个文本
    processed = preprocessor.process_text(sample_text)

    print("原始文本:", sample_text)
    print("\n清洗后的文本:", processed["cleaned_text"])
    print("\n标准化后的文本:", processed["normalized_text"])
    print("\n分词结果:", processed["tokens"])
    print("\n数值特征:", {k: v for k, v in processed.items() if k.startswith("numeric_")})
    print("\nWeb3特征:", {k: v for k, v in processed.items() if k.startswith("web3_")})

    # 如果有分词器，显示编码结果
    if "input_ids" in processed:
        print("\n编码后的输入ID:", processed["input_ids"][:10], "...")
        print("注意力掩码:", processed["attention_mask"][:10], "...")

    # 示例数据集
    sample_dataset = [
        {
            "id": 1,
            "text": "如何在以太坊上部署智能合约？",
            "label": 0
        },
        {
            "id": 2,
            "text": "比特币价格预测：未来一周BTC会继续上涨吗？",
            "label": 1
        },
        {
            "id": 3,
            "text": "这是一个关于DAO治理的讨论，欢迎参与！",
            "label": 0
        }
    ]

    # 创建数据集处理器
    dataset_processor = Web3DatasetProcessor(preprocessor)

    # 处理数据集
    processed_dataset = dataset_processor.process_dataset(sample_dataset)

    print("\n处理后的数据集:")
    for item in processed_dataset:
        print(f"ID: {item['id']}, 标签: {item.get('label')}")
        print(f"清洗后的文本: {item['cleaned_text']}")
        print(f"Web3术语总数: {item['web3_term_total']}")
        print("-" * 40)

    # 转换为DataFrame
    df = pd.DataFrame(sample_dataset)
    processed_df = dataset_processor.process_dataframe(df)

    print("\n处理后的DataFrame:")
    print(processed_df[["id", "label", "cleaned_text", "web3_term_total"]])

    # 创建PyTorch数据集
    torch_dataset = dataset_processor.create_torch_dataset(processed_dataset)

    print("\nPyTorch数据集样本:")
    sample = torch_dataset[0]
    for key, value in sample.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value[:10]}")