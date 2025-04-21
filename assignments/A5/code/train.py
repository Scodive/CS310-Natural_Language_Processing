import sys
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from get_train_data import FeatureExtractor
from model import BaseModel, WordPOSModel
from bilstm_model import BiLSTMWordPOSModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', default='input_train.npy')
argparser.add_argument('--target_file', default='target_train.npy')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')
argparser.add_argument('--model', default=None, help='path to save model file, if not specified, a .pt with timestamp will be used')
argparser.add_argument('--model_type', choices=['base', 'wordpos', 'bilstm'], default='wordpos', help='type of model to train')

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
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.rel_vocab)

    ### START YOUR CODE ###
    # 根据选择初始化不同的模型
    print(f"初始化{args.model_type}模型...")
    if args.model_type == 'base':
        model = BaseModel(word_vocab_size, output_size)
    elif args.model_type == 'wordpos':
        model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    elif args.model_type == 'bilstm':
        model = BiLSTMWordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    ### END YOUR CODE ###

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.NLLLoss()

    # 加载数据
    inputs = np.load(args.input_file)
    targets = np.load(args.target_file)
    print(f"加载数据完成。输入形状: {inputs.shape}, 目标形状: {targets.shape}")

    # 划分训练集和验证集
    num_samples = len(inputs)
    indices = np.random.permutation(num_samples)
    val_size = int(0.1 * num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]

    # 转换为PyTorch张量
    train_inputs = torch.from_numpy(train_inputs).float()
    train_targets = torch.from_numpy(train_targets).long()
    val_inputs = torch.from_numpy(val_inputs).float()
    val_targets = torch.from_numpy(val_targets).long()

    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=4096)

    # 训练参数
    n_epochs = 10
    batch_size = 4096
    print_loss_every = 100
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 训练循环
    print(f"开始训练{args.model_type}模型...")
    for epoch in range(n_epochs):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            inputs_batch, targets_batch = batch
            
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_count % print_loss_every == 0:
                avg_loss = train_loss / batch_count 
                sys.stdout.write(f'\rEpoch {epoch+1}/{n_epochs} - Batch {batch_count}/{len(train_loader)} - Loss: {avg_loss:.4f}')
                sys.stdout.flush()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_targets = val_batch
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
                
                _, predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += predicted.eq(val_targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{n_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Time: {epoch_time:.2f} sec')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 保存模型
    if args.model is not None:
        model_path = args.model
    else:
        # 为不同类型的模型设置默认保存路径
        default_paths = {
            'base': 'model.pt',
            'wordpos': 'model.pt',
            'bilstm': 'bilstm.pt'
        }
        model_path = default_paths[args.model_type]
    
    print(f"保存模型到 {model_path}")
    torch.save(model.state_dict(), model_path)