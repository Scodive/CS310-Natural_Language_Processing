import sys
import copy
from collections import defaultdict
from typing import List, Tuple


class DependencyArc(object):
    """
    Represent a single dependency arc:
    """
    def __init__(self, word_id, word, pos, head, deprel):
        self.id = word_id
        self.word = word
        self.pos = pos
        self.head = head
        self.deprel = deprel
    
    def __str__(self) -> str:
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split("\t")
    word_id_str, word, lemma, upos, pos, feats, head_str, deprel, deps, misc = fields
    word_id = int(word_id_str)
    head = int(head_str)
    return DependencyArc(word_id, word, pos, head, deprel)


class DependencyTree(object):
    def __init__(self):
        self.deprels = {}
        self.root = None
        self.parent_to_children = defaultdict(list)

    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self):
        deprels = [v for (k, v) in sorted(self.deprels.items())]
        return "\n".join(str(deprel) for deprel in deprels)
    
    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent:
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return "({} {})".format(self.deprels[parent].word, child_str)

    def words(self):
        return [None] + [x.word for (i, x) in self.deprels.items()]

    def pos(self):
        return [None] + [x.pos for (i, x) in self.deprels.items()]
    
    def from_string(s):
        dtree = DependencyTree()
        for line in s.split("\n"):
            if line:
                dtree.add_deprel(parse_conll_relation(line))
        return dtree


def conll_reader(input_file):
    current_deps = DependencyTree()
    while True:
        line = input_file.readline().strip()
        if not line and current_deps:
            yield current_deps
            current_deps = DependencyTree()
            line = input_file.readline().strip()
            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line))


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()
    
    def shift(self):
        """Arc-eager: 将buffer顶部的词移到stack顶部"""
        self.stack.append(self.buffer.pop())

    def reduce(self):
        """Arc-eager: 弹出stack顶部的词（当且仅当它已经有head）"""
        self.stack.pop()

    def right_arc(self, label):
        """Arc-eager: 添加一个从stack顶部指向buffer顶部的弧，并将buffer顶部移到stack"""
        parent = self.stack[-1]
        child = self.buffer[-1]
        self.deps.add((parent, child, label))
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        """Arc-eager: 添加一个从buffer顶部指向stack顶部的弧，并弹出stack顶部"""
        parent = self.buffer[-1]
        child = self.stack.pop()
        self.deps.add((parent, child, label))

    def is_final(self):
        """检查是否达到终止状态"""
        return len(self.buffer) == 0 and len(self.stack) <= 1

    def can_reduce(self):
        """检查是否可以执行reduce操作"""
        if not self.stack:
            return False
        stack_top = self.stack[-1]
        # 检查stack顶部的词是否已经有head
        return any(head == stack_top or dep == stack_top for (head, dep, _) in self.deps)

    def can_left_arc(self):
        """检查是否可以执行left_arc操作"""
        if not self.stack or not self.buffer:
            return False
        stack_top = self.stack[-1]
        # stack顶部的词不能已经有head
        return not any(head == stack_top or dep == stack_top for (head, dep, _) in self.deps)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_tree: DependencyTree) -> List[Tuple[State, Tuple[str, str]]]:
    deprels = dep_tree.deprels
    sorted_nodes = [k for k, v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident, node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]

        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id] -= 1
            seq.append((copy.deepcopy(state), ("left_arc", stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id] -= 1
            seq.append((copy.deepcopy(state), ("right_arc", bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()

    return seq