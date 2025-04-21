import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        # 定义模型架构
        self.embedding = nn.Embedding(word_vocab_size, 100)  # 词嵌入层
        self.hidden1 = nn.Linear(600, 200)  # 第一个隐藏层
        self.hidden2 = nn.Linear(200, 100)  # 第二个隐藏层
        self.output = nn.Linear(100, output_size)  # 输出层
        
        # 激活函数
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        ### END YOUR CODE ###
    
    def forward(self, x):
        ### START YOUR CODE ###
        # x的形状为 [batch_size, 6]
        embedded = self.embedding(x.long())  # [batch_size, 6, 100]
        flattened = embedded.view(embedded.shape[0], -1)  # [batch_size, 600]
        
        # 前向传播
        hidden1 = self.relu(self.hidden1(flattened))
        hidden2 = self.relu(self.hidden2(hidden1))
        output = self.log_softmax(self.output(hidden2))
        ### END YOUR CODE ###
        return output


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        ### START YOUR CODE ###
        # 定义模型架构
        self.word_embedding_dim = 50
        self.pos_embedding_dim = 50
        
        # 嵌入层
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, self.pos_embedding_dim)
        
        # 计算输入维度：6个位置，每个位置有词和词性特征
        self.feature_dim = self.word_embedding_dim + self.pos_embedding_dim  # 50 + 50 = 100
        self.input_dim = 6 * self.feature_dim  # 6 * 100 = 600
        
        # 全连接层
        self.hidden1 = nn.Linear(self.input_dim, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, output_size)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(200)
        self.layer_norm2 = nn.LayerNorm(100)
        self.log_softmax = nn.LogSoftmax(dim=1)
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # 检查输入维度
        if x.dim() != 2 or x.size(1) != 12:
            raise ValueError(f"Expected input shape [batch_size, 12], got {x.shape}")
            
        batch_size = x.size(0)
        
        # 分离词和词性特征
        word_ids = x[:, :6].long()  # [batch_size, 6]
        pos_ids = x[:, 6:].long()   # [batch_size, 6]
        
        # 检查ID的范围
        if torch.any(word_ids < 0) or torch.any(pos_ids < 0):
            raise ValueError("Found negative IDs in input")
            
        # 嵌入
        word_embedded = self.word_embedding(word_ids)  # [batch_size, 6, word_emb_dim]
        pos_embedded = self.pos_embedding(pos_ids)     # [batch_size, 6, pos_emb_dim]
        
        # 连接每个位置的特征
        word_embedded = word_embedded.view(batch_size, 6, -1)  # 确保形状正确
        pos_embedded = pos_embedded.view(batch_size, 6, -1)
        features = torch.cat([word_embedded, pos_embedded], dim=2)  # [batch_size, 6, feature_dim]
        
        # 展平特征
        features = features.view(batch_size, -1)  # [batch_size, input_dim]
        
        # 前向传播
        hidden1 = self.relu(self.layer_norm1(self.hidden1(features)))
        hidden1 = self.dropout(hidden1)
        
        hidden2 = self.relu(self.layer_norm2(self.hidden2(hidden1)))
        hidden2 = self.dropout(hidden2)
        
        output = self.log_softmax(self.output(hidden2))
        ### END YOUR CODE ###
        return output
