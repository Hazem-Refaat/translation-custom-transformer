import torch 
import torch.nn as nn
import math
# Traslate arabic to english Transformer
"""Positional Encoder"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, input_q, input_k, input_v, mask=None):
        batch_size = input_q.size(0)

        q = self.W_q(input_q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(input_k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(input_v).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)# toch.matmul is Matrix product of two tensors.

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        output = self.fc(output)

        return output

# Implement position-wise feed-forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Implement encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.feed_forward_network(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        attn_output = self.masked_multi_head_attention(x, x, x, tgt_mask)
        attn_output = self.dropout(attn_output)
        out1 = self.layernorm1(x + attn_output)

        attn_output = self.encoder_decoder_attention(out1, encoder_output, encoder_output, src_mask)
        attn_output = self.dropout(attn_output)
        out2 = self.layernorm2(out1 + attn_output)

        ffn_output = self.feed_forward_network(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
    
# Implement encoder
class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, input_vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, output_vocab_size, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, tgt_vocab_size, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output