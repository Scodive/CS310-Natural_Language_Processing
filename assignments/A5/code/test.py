import sys
import os
import argparse
from collections import defaultdict
from typing import List, Tuple
import torch

from parse_utils import conll_reader, DependencyTree
from parser import Parser
from get_train_data import FeatureExtractor

def compute_metrics(gold_trees: List[DependencyTree], pred_trees: List[DependencyTree]) -> Tuple[float, float, float]:
    """计算UAS（无标记依存准确率）和LAS（有标记依存准确率）"""
    total_words = 0
    correct_uas = 0
    correct_las = 0
    
    for gold_tree, pred_tree in zip(gold_trees, pred_trees):
        for word_id in gold_tree.deprels:
            if word_id == 0:  # 跳过ROOT
                continue
            total_words += 1
            
            # 获取预测和真实的头节点和关系
            pred_head = pred_tree.deprels[word_id].head if word_id in pred_tree.deprels else None
            gold_head = gold_tree.deprels[word_id].head
            
            pred_rel = pred_tree.deprels[word_id].deprel if word_id in pred_tree.deprels else None
            gold_rel = gold_tree.deprels[word_id].deprel
            
            # 计算UAS
            if pred_head == gold_head:
                correct_uas += 1
                # 计算LAS
                if pred_rel == gold_rel:
                    correct_las += 1
    
    # 计算准确率
    uas = correct_uas / total_words if total_words > 0 else 0
    las = correct_las / total_words if total_words > 0 else 0
    
    return uas, las, total_words

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('test_file', help='Path to the test file')
    argparser.add_argument('--model', type=str, default='model.pt', help='Path to the trained model')
    argparser.add_argument('--words_vocab', default='words_vocab.txt', help='Path to words vocabulary')
    argparser.add_argument('--pos_vocab', default='pos_vocab.txt', help='Path to POS vocabulary')
    argparser.add_argument('--rel_vocab', default='rel_vocab.txt', help='Path to relation vocabulary')
    argparser.add_argument('--output', default='output.conll', help='Path to output predictions')
    args = argparser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.test_file):
        print(f"测试文件 {args.test_file} 不存在")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"模型文件 {args.model} 不存在")
        sys.exit(1)

    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError as e:
        print(f"词汇表文件不存在: {e}")
        sys.exit(1)

    # 初始化特征提取器和解析器
    print("初始化解析器...")
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    # 读取测试数据
    print("读取测试数据...")
    gold_trees = []
    pred_trees = []
    with open(args.test_file, 'r') as f:
        for gold_tree in conll_reader(f):
            gold_trees.append(gold_tree)
            words = gold_tree.words()
            pos = gold_tree.pos()
            pred_tree = parser.parse_sentence(words, pos)
            pred_trees.append(pred_tree)

    # 计算性能指标
    print("计算性能指标...")
    uas, las, total_words = compute_metrics(gold_trees, pred_trees)
    print(f"测试结果 (共 {total_words} 个词):")
    print(f"UAS (无标记依存准确率): {uas:.4f}")
    print(f"LAS (有标记依存准确率): {las:.4f}")

    # 保存预测结果
    print(f"保存预测结果到 {args.output}...")
    with open(args.output, 'w') as f:
        for tree in pred_trees:
            f.write(str(tree))
            f.write("\n\n")

if __name__ == "__main__":
    main() 