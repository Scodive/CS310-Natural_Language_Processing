import torch
import torch.nn as nn
import math

from config import *


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)  # token embedding
        # self.pos_embed = nn.Embedding(MAX_LEN, d_model)  # position embedding
        self.pos_embed = PositionalEmbedding(d_model, max_len=MAX_LEN)
        self.seg_embed = nn.Embedding(n_segments + 1, d_model, padding_idx=0)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k) # scores : [batch_size x n_heads x seq_len x =seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        # x: [batch_size x seq_len x d_model]
        residual, batch_size = x, x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x seq_len x d_k]
        k_s = self.W_K(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x seq_len x d_k]
        v_s = self.W_V(x).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x seq_len x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x seq_len x seq_len]

        # context: [batch_size x n_heads x seq_len x d_v], attn: [batch_size x n_heads x seq_len x seq_len]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x seq_len x n_heads * d_v]
        output = self.fc(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x seq_len x d_model]


# Activation function: GELU (Gaussian Error Linear Unit)
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_pad_attn_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=seq_len), one is masking
    return pad_attn_mask.expand(batch_size, seq_len, len_k)  # batch_size x seq_len x len_k


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        ### START YOUR CODE ###
        # Call enc_self_attn and pos_ffn
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        ### END YOUR CODE ###
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

         # for NSP task
        self.fc1 = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        # for MLM task
        self.fc2 = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight # decoder is shared with embedding layer
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_pad_attn_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]

        # Use the representation of [CLS] to produce logits for NSP task
        # First, gather the representations of [CLS] token, then pass it to self.classifier
        ### START YOUR CODE ###
        h_pooled = output[:, 0] # Gather [CLS] token representation
        h_pooled = self.activ1(self.fc1(h_pooled))
        logits_clsf = self.classifier(h_pooled)
        ### END YOUR CODE ###

        # Gather the representations of masked tokens to produce logits for MLM task
        # Hint: use masked_pos to select masked tokens; use torch.gather to gather the representations
        ### START YOUR CODE ###
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # expand dimension for gather
        h_masked = torch.gather(output, 1, masked_pos) # gather masked tokens
        h_masked = self.norm(self.activ2(self.fc2(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        ### END YOUR CODE ###

        return logits_lm, logits_clsf