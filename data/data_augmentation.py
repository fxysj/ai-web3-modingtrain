# 数据增强工具
# utils/data_augmentation.py
import random
import re
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from nltk.corpus import wordnet
import nltk
from collections import defaultdict

# 确保nltk数据已下载
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class Web3DataAugmenter:
    """Web3.0领域文本数据增强工具"""

    def __init__(self, model_name: str = "bert-base-chinese", seed: int = 42):
        """
        初始化数据增强器

        Args:
            model_name: 用于掩码语言模型的预训练模型名称
            seed: 随机数种子
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.mlm_pipeline = pipeline("fill-mask", model=model_name, tokenizer=model_name)
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Web3.0领域术语表
        self.web3_terms = {
            "加密货币": ["数字货币", "虚拟货币", "加密资产"],
            "智能合约": ["自执行合约", "区块链合约"],
            "去中心化": ["分布式", "去中介化"],
            "NFT": ["非同质化代币", "不可替代代币"],
            "DeFi": ["去中心化金融", "开放式金融"],
            "DAO": ["去中心化自治组织", "分布式自治组织"],
            "质押": ["抵押", "锁定", "Staking"],
            "流动性": ["资金流动性", "市场流动性"],
            "预言机": ["Oracle", "数据预言机"],
            "Gas费": ["燃料费", "交易费"],
            "钱包": ["数字钱包", "加密钱包"],
            "私钥": ["私密密钥", "个人密钥"],
            "公钥": ["公开密钥", "共享密钥"],
            "哈希": ["散列", "Hash值"],
            "挖矿": ["区块挖掘", "加密货币挖掘"],
            "区块链": ["分布式账本", "区块链接"],
            "以太坊": ["Ethereum", "ETH网络"],
            "比特币": ["Bitcoin", "BTC"],
            "稳定币": ["加密稳定币", "算法稳定币"]
        }

        # 反义词词典，用于否定增强
        self.antonyms = {
            "上涨": ["下跌", "下降", "回落"],
            "增加": ["减少", "降低", "缩减"],
            "买入": ["卖出", "抛售", "沽出"],
            "多头": ["空头", "看空"],
            "牛市": ["熊市", "空头市场"],
            "盈利": ["亏损", "损失", "蚀本"],
            "高": ["低", "矮"],
            "快": ["慢", "缓慢"],
            "新": ["旧", "老", "过时"],
            "安全": ["危险", "风险", "不安全"],
            "合法": ["非法", "违规", "不合法"]
        }

        # 正则表达式模式，用于识别数值和百分比
        self.number_pattern = re.compile(r'\d+(\.\d+)?')
        self.percentage_pattern = re.compile(r'\d+(\.\d+)?%')

    def synonym_replacement(self, text: str, n: int = 3) -> List[str]:
        """
        同义词替换增强

        Args:
            text: 输入文本
            n: 生成的增强样本数量

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            words = text.split()
            new_words = []

            for word in words:
                # 优先使用Web3.0领域术语替换
                if word in self.web3_terms:
                    synonyms = self.web3_terms[word]
                    new_words.append(random.choice(synonyms) if synonyms else word)
                else:
                    # 使用WordNet查找同义词
                    synsets = wordnet.synsets(word)
                    if synsets:
                        synonyms = []
                        for syn in synsets:
                            for lemma in syn.lemmas():
                                if lemma.name() != word:
                                    synonyms.append(lemma.name())

                        if synonyms:
                            new_words.append(random.choice(synonyms))
                        else:
                            new_words.append(word)
                    else:
                        new_words.append(word)

            augmented_texts.append(" ".join(new_words))

        return augmented_texts

    def antonym_replacement(self, text: str, n: int = 3) -> List[str]:
        """
        反义词替换增强（用于生成对比样本）

        Args:
            text: 输入文本
            n: 生成的增强样本数量

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            words = text.split()
            new_words = []
            modified = False

            for word in words:
                if word in self.antonyms:
                    antonyms = self.antonyms[word]
                    new_words.append(random.choice(antonyms))
                    modified = True
                else:
                    new_words.append(word)

            # 只有当文本被修改时才添加
            if modified:
                augmented_texts.append(" ".join(new_words))

        return augmented_texts

    def insertion_with_mlm(self, text: str, n: int = 3, num_new_words: int = 2) -> List[str]:
        """
        使用掩码语言模型插入新词

        Args:
            text: 输入文本
            n: 生成的增强样本数量
            num_new_words: 每个样本插入的新词数量

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            words = text.split()
            if not words:
                continue

            # 随机选择位置插入[MASK]
            for _ in range(num_new_words):
                if len(words) == 0:
                    break

                mask_index = random.randint(0, len(words))
                words.insert(mask_index, self.mlm_pipeline.tokenizer.mask_token)

            # 用掩码语言模型填充[MASK]
            masked_text = " ".join(words)
            try:
                filled_texts = self.mlm_pipeline(masked_text, top_k=1)
                if isinstance(filled_texts, list) and len(filled_texts) > 0:
                    filled_text = filled_texts[0]["sequence"]
                    augmented_texts.append(filled_text)
            except Exception as e:
                print(f"Error in MLM insertion: {e}")
                augmented_texts.append(text)  # 出错时返回原始文本

        return augmented_texts

    def deletion(self, text: str, n: int = 3, p: float = 0.1) -> List[str]:
        """
        随机删除单词

        Args:
            text: 输入文本
            n: 生成的增强样本数量
            p: 每个单词被删除的概率

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            words = text.split()
            if not words:
                augmented_texts.append("")
                continue

            new_words = [word for word in words if random.random() > p]

            # 确保至少保留一个单词
            if not new_words:
                new_words = [random.choice(words)]

            augmented_texts.append(" ".join(new_words))

        return augmented_texts

    def swap_words(self, text: str, n: int = 3, p: float = 0.1) -> List[str]:
        """
        随机交换相邻单词

        Args:
            text: 输入文本
            n: 生成的增强样本数量
            p: 每个位置进行交换的概率

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            words = text.split()
            if len(words) < 2:
                augmented_texts.append(text)
                continue

            new_words = words.copy()
            num_swaps = int(p * len(words))

            for _ in range(num_swaps):
                if len(new_words) < 2:
                    break

                idx = random.randint(0, len(new_words) - 2)
                new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]

            augmented_texts.append(" ".join(new_words))

        return augmented_texts

    def numerical_variation(self, text: str, n: int = 3, variation_range: float = 0.1) -> List[str]:
        """
        对数值进行小范围变化（适用于价格、数量等数值）

        Args:
            text: 输入文本
            n: 生成的增强样本数量
            variation_range: 数值变化范围（百分比）

        Returns:
            增强后的文本列表
        """
        augmented_texts = []

        for _ in range(n):
            new_text = text

            # 处理百分比
            for match in self.percentage_pattern.finditer(new_text):
                value = float(match.group(0)[:-1])  # 去掉%符号
                # 生成±variation_range范围内的随机变化
                variation = value * random.uniform(-variation_range, variation_range)
                new_value = value + variation
                new_text = new_text.replace(match.group(0), f"{new_value:.2f}%")

            # 处理普通数值
            for match in self.number_pattern.finditer(new_text):
                value = float(match.group(0))
                # 生成±variation_range范围内的随机变化
                variation = value * random.uniform(-variation_range, variation_range)
                new_value = value + variation

                # 根据原始数值格式决定保留的小数位数
                if '.' in match.group(0):
                    decimal_places = len(match.group(0).split('.')[1])
                    new_text = new_text.replace(match.group(0), f"{new_value:.{decimal_places}f}")
                else:
                    new_text = new_text.replace(match.group(0), f"{int(round(new_value))}")

            augmented_texts.append(new_text)

        return augmented_texts

    def combine_methods(self, text: str, n: int = 5, methods: Optional[List[str]] = None) -> List[str]:
        """
        组合多种增强方法

        Args:
            text: 输入文本
            n: 生成的增强样本数量
            methods: 要使用的增强方法列表，默认为所有方法

        Returns:
            增强后的文本列表
        """
        if not methods:
            methods = [
                "synonym_replacement",
                "insertion_with_mlm",
                "deletion",
                "swap_words",
                "numerical_variation"
            ]

        augmented_texts = [text]  # 初始包含原始文本

        for _ in range(n):
            # 随机选择一种增强方法
            method = random.choice(methods)

            if method == "synonym_replacement":
                new_texts = self.synonym_replacement(text, n=1)
            elif method == "antonym_replacement":
                new_texts = self.antonym_replacement(text, n=1)
            elif method == "insertion_with_mlm":
                new_texts = self.insertion_with_mlm(text, n=1)
            elif method == "deletion":
                new_texts = self.deletion(text, n=1)
            elif method == "swap_words":
                new_texts = self.swap_words(text, n=1)
            elif method == "numerical_variation":
                new_texts = self.numerical_variation(text, n=1)
            else:
                continue

            if new_texts:
                augmented_texts.append(new_texts[0])

        # 移除重复项并排除原始文本
        unique_texts = list(set(augmented_texts))
        if text in unique_texts and len(unique_texts) > 1:
            unique_texts.remove(text)

        return unique_texts[:n]  # 确保返回n个样本


class Web3DataPairAugmenter:
    """Web3.0领域文本对数据增强工具（适用于问答、对比等任务）"""

    def __init__(self, base_augmenter: Web3DataAugmenter = None, seed: int = 42):
        """
        初始化文本对增强器

        Args:
            base_augmenter: 基础增强器实例
            seed: 随机数种子
        """
        self.base_augmenter = base_augmenter or Web3DataAugmenter(seed=seed)
        self.seed = seed
        random.seed(seed)

    def augment_qa_pair(self, question: str, answer: str, n: int = 5) -> List[Tuple[str, str]]:
        """
        增强问答对

        Args:
            question: 问题文本
            answer: 回答文本
            n: 生成的增强样本数量

        Returns:
            增强后的问答对列表
        """
        augmented_pairs = []

        # 分别增强问题和回答
        augmented_questions = self.base_augmenter.combine_methods(question, n=n)
        augmented_answers = self.base_augmenter.combine_methods(answer, n=n)

        # 生成新的问答对
        for q, a in zip(augmented_questions, augmented_answers):
            augmented_pairs.append((q, a))

        # 随机组合问题和回答（增加多样性）
        if len(augmented_questions) > 1 and len(augmented_answers) > 1:
            for _ in range(min(n, 3)):  # 添加几个随机组合
                random_q = random.choice(augmented_questions)
                random_a = random.choice(augmented_answers)
                augmented_pairs.append((random_q, random_a))

        return augmented_pairs[:n]  # 确保返回n个样本

    def augment_comparison_pair(self, text1: str, text2: str, n: int = 5) -> List[Tuple[str, str]]:
        """
        增强对比文本对（如正面和负面观点）

        Args:
            text1: 第一个文本
            text2: 第二个文本
            n: 生成的增强样本数量

        Returns:
            增强后的文本对列表
        """
        augmented_pairs = []

        # 分别增强两个文本
        augmented_texts1 = self.base_augmenter.combine_methods(text1, n=n)
        augmented_texts2 = self.base_augmenter.combine_methods(text2, n=n)

        # 保持对比关系的增强
        for t1, t2 in zip(augmented_texts1, augmented_texts2):
            augmented_pairs.append((t1, t2))

        # 使用反义词替换增强对比性
        antonym_pairs = self.base_augmenter.antonym_replacement(text1, n=min(n, 2))
        for antonym_text in antonym_pairs:
            # 为反义词文本找到合适的配对
            paired_text = random.choice(augmented_texts2)
            augmented_pairs.append((antonym_text, paired_text))

        return augmented_pairs[:n]  # 确保返回n个样本


def augment_dataset(data: List[Dict[str, Any]], augmenter: Web3DataAugmenter,
                    augment_field: str = "text", n_per_sample: int = 3) -> List[Dict[str, Any]]:
    """
    增强整个数据集

    Args:
        data: 原始数据集，每个元素是一个字典
        augmenter: 数据增强器
        augment_field: 需要增强的字段名
        n_per_sample: 每个样本生成的增强样本数量

    Returns:
        增强后的数据集
    """
    augmented_data = []

    for item in data:
        # 确保数据项包含需要增强的字段
        if augment_field not in item:
            augmented_data.append(item)
            continue

        original_text = item[augment_field]

        # 生成增强样本
        augmented_texts = augmenter.combine_methods(original_text, n=n_per_sample)

        # 为每个增强样本创建新的数据项
        for text in augmented_texts:
            new_item = item.copy()
            new_item[augment_field] = text
            new_item["is_augmented"] = True  # 添加标记，表示这是增强样本
            augmented_data.append(new_item)

    # 添加原始数据
    for item in data:
        item["is_augmented"] = False  # 标记为原始样本
        augmented_data.append(item)

    return augmented_data


# 示例使用
if __name__ == "__main__":
    # 创建增强器
    augmenter = Web3DataAugmenter()

    # 示例文本
    web3_text = "以太坊是一个开源的有智能合约功能的公共区块链平台，支持去中心化应用开发。"

    # 测试各种增强方法
    print("原始文本:", web3_text)
    print("\n同义词替换:")
    for text in augmenter.synonym_replacement(web3_text, n=3):
        print("-", text)

    print("\n反义词替换:")
    for text in augmenter.antonym_replacement("比特币价格上涨可能导致更多投资者买入", n=3):
        print("-", text)

    print("\nMLM插入:")
    for text in augmenter.insertion_with_mlm(web3_text, n=3):
        print("-", text)

    print("\n随机删除:")
    for text in augmenter.deletion(web3_text, n=3):
        print("-", text)

    print("\n单词交换:")
    for text in augmenter.swap_words(web3_text, n=3):
        print("-", text)

    print("\n数值变化:")
    price_text = "比特币价格为42,000美元，较昨日上涨3.25%。"
    for text in augmenter.numerical_variation(price_text, n=3):
        print("-", text)

    print("\n组合增强:")
    for text in augmenter.combine_methods(web3_text, n=3):
        print("-", text)

    # 测试问答对增强
    pair_augmenter = Web3DataPairAugmenter(augmenter)
    question = "什么是DeFi协议？"
    answer = "DeFi协议是基于区块链的金融应用，提供传统金融服务的去中心化版本。"

    print("\n问答对增强:")
    for q, a in pair_augmenter.augment_qa_pair(question, answer, n=3):
        print(f"Q: {q}")
        print(f"A: {a}")
        print("-" * 40)