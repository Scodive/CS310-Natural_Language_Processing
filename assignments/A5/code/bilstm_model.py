import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMWordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size, 
                 word_embedding_dim=100, pos_embedding_dim=50, 
                 hidden_dim=200, num_layers=2, dropout=0.5):
        super(BiLSTMWordPOSModel, self).__init__()
        
        # 嵌入层
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        
        # 嵌入层dropout
        self.embed_dropout = nn.Dropout(0.2)
        
        # BiLSTM层
        self.input_dim = word_embedding_dim + pos_embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差连接和层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 4)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 分离词和词性特征
        word_ids = x[:, :6].long()
        pos_ids = x[:, 6:].long()
        
        # 嵌入并应用dropout
        word_embedded = self.embed_dropout(self.word_embedding(word_ids))
        pos_embedded = self.embed_dropout(self.pos_embedding(pos_ids))
        
        # 连接词和词性嵌入
        embedded = torch.cat([word_embedded, pos_embedded], dim=2)
        
        # BiLSTM编码
        lstm_out, _ = self.lstm(embedded)
        
        # 残差连接和层归一化
        lstm_out = self.layer_norm1(lstm_out)
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = lstm_out + attn_out  # 残差连接
        attn_out = self.layer_norm2(attn_out)
        
        # 提取stack和buffer顶部的表示
        stack_top = attn_out[:, 0, :]
        buffer_top = attn_out[:, -1, :]
        
        # 连接特征
        combined = torch.cat([stack_top, buffer_top], dim=1)
        
        # MLP分类器
        hidden = self.dropout(F.gelu(self.fc1(combined)))
        hidden = self.layer_norm3(hidden)
        hidden = self.dropout(F.gelu(self.fc2(hidden)))
        output = self.fc3(hidden)
        
        return F.log_softmax(output, dim=1)

class BiLSTMFeatureExtractor:
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        self.model = BiLSTMWordPOSModel(word_vocab_size, pos_vocab_size, output_size)
        
    def extract_features(self, words, pos, state):
        """从当前状态提取特征"""
        # 获取stack和buffer中的词和它们的上下文
        stack_indices = state.stack[-3:] if len(state.stack) > 0 else []
        buffer_indices = state.buffer[-3:] if len(state.buffer) > 0 else []
        
        # 获取词和词性，包括上下文信息
        stack_words = []
        stack_pos = []
        for idx in stack_indices:
            if 0 <= idx < len(words):
                # 添加当前词及其上下文
                stack_words.append(words[idx])
                stack_pos.append(pos[idx])
            else:
                stack_words.append("<NULL>")
                stack_pos.append("<NULL>")
        
        buffer_words = []
        buffer_pos = []
        for idx in buffer_indices:
            if 0 <= idx < len(words):
                # 添加当前词及其上下文
                buffer_words.append(words[idx])
                buffer_pos.append(pos[idx])
            else:
                buffer_words.append("<NULL>")
                buffer_pos.append("<NULL>")
        
        # 填充到固定长度
        while len(stack_words) < 3:
            stack_words.append("<NULL>")
            stack_pos.append("<NULL>")
        while len(buffer_words) < 3:
            buffer_words.append("<NULL>")
            buffer_pos.append("<NULL>")
        
        return stack_words + buffer_words, stack_pos + buffer_pos 