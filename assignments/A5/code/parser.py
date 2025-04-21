import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel
from parse_utils import DependencyArc, DependencyTree, State, parse_conll_relation
from get_train_data import FeatureExtractor

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.pt')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')


class Parser(object):
    def __init__(self, extractor: FeatureExtractor, model_file: str):
        ### START YOUR CODE ###
        # 初始化模型
        word_vocab_size = len(extractor.word_vocab)
        pos_vocab_size = len(extractor.pos_vocab)
        output_size = len(extractor.rel_vocab)
        self.model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
        ### END YOUR CODE ###
        self.model.load_state_dict(torch.load(model_file, weights_only=True))
        self.model.eval()  # 设置为评估模式
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.rel_vocab.items()]
        )

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)  # 添加ROOT

        while not state.is_final():
            ### START YOUR CODE ###
            # 获取当前状态的特征表示
            current_state = self.extractor.get_input_repr_wordpos(words, pos, state)
            
            # 转换为tensor并添加batch维度
            with torch.no_grad():
                input_tensor = torch.tensor(current_state).float().unsqueeze(0)
                prediction = self.model(input_tensor)
                best_action_idx = prediction.argmax(dim=1).item()
                best_action = self.output_labels[best_action_idx]
            
            # Arc-eager转换系统
            action_type = best_action[0]
            
            # 检查是否可以执行当前动作
            if action_type == "shift" and state.buffer:
                state.shift()
            elif action_type == "reduce" and state.can_reduce():
                state.reduce()
            elif action_type == "left_arc" and state.can_left_arc():
                state.left_arc(best_action[1])
            elif action_type == "right_arc" and state.buffer:
                state.right_arc(best_action[1])
            else:
                # 如果当前动作不可执行，选择一个默认动作
                if state.buffer:
                    state.shift()
                elif state.can_reduce():
                    state.reduce()
                else:
                    break
            ### END YOUR CODE ###

        ### START YOUR CODE ###
        # 构建依存树
        tree = DependencyTree()
        
        # 添加所有依存关系
        for head, dependent, relation in state.deps:
            deprel = DependencyArc(
                word_id=dependent,
                word=words[dependent],
                pos=pos[dependent],
                head=head,
                deprel=relation
            )
            tree.add_deprel(deprel)
        
        # 找到并添加根节点
        root_id = None
        for i in range(1, len(words)):
            if all(dep[1] != i for dep in state.deps):
                root_id = i
                break
        
        if root_id is not None:
            root_deprel = DependencyArc(
                word_id=root_id,
                word=words[root_id],
                pos=pos[root_id],
                head=0,
                deprel="root"
            )
            tree.add_deprel(root_deprel)
        ### END YOUR CODE ###

        return tree

if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)
    
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    # Test an example sentence, 3rd example from dev.conll
    words = [None, 'The', 'bill', 'intends', 'to', 'restrict', 'the', 'RTC', 'to', 'Treasury', 'borrowings', 'only', ',', 'unless', 'the', 'agency', 'receives', 'specific', 'congressional', 'authorization', '.']
    pos = [None, 'DT', 'NN', 'VBZ', 'TO', 'VB', 'DT', 'NNP', 'TO', 'NNP', 'NNS', 'RB', ',', 'IN', 'DT', 'NN', 'VBZ', 'JJ', 'JJ', 'NN', '.']

    tree = parser.parse_sentence(words, pos)
    print(tree)