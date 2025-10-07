from torch import nn
import torch
import math

class MLPModel(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer模型用于时间序列预测"""
    
    def __init__(self, 
                 input_size=8,           # 输入特征维度
                 d_model=64,             # 模型维度
                 nhead=8,                # 注意力头数
                 num_layers=3,           # Transformer层数
                 dim_feedforward=256,    # 前馈网络维度
                 dropout=0.1,            # dropout率
                 seq_len=20,             # 序列长度
                 output_size=1):         # 输出维度
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
        Returns:
            输出张量，形状为 (batch_size, output_size)
        """
        batch_size, seq_len, input_size = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        # 生成注意力掩码（避免未来信息泄露）
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer_encoder(x, mask=mask)  # (batch_size, seq_len, d_model)
        
        # 展平并输出
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * d_model)
        output = self.output_layer(x)  # (batch_size, output_size)
        
        return output.squeeze(-1)
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果注意力掩码，避免未来信息泄露"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerWithDecoder(nn.Module):
    """带解码器的Transformer模型，用于多步预测"""
    
    def __init__(self, 
                 input_size=8,           # 输入特征维度
                 d_model=64,             # 模型维度
                 nhead=8,                # 注意力头数
                 num_encoder_layers=3,   # 编码器层数
                 num_decoder_layers=3,   # 解码器层数
                 dim_feedforward=256,    # 前馈网络维度
                 dropout=0.1,            # dropout率
                 seq_len=20,             # 序列长度
                 pred_len=5,             # 预测长度
                 output_size=1):         # 输出维度
        super(TransformerWithDecoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None):
        """
        前向传播
        Args:
            src: 源序列，形状为 (batch_size, seq_len, input_size)
            tgt: 目标序列（训练时使用），形状为 (batch_size, pred_len, input_size)
        Returns:
            预测序列，形状为 (batch_size, pred_len, output_size)
        """
        batch_size = src.size(0)
        
        # 源序列处理
        src_proj = self.input_projection(src)  # (batch_size, seq_len, d_model)
        src_proj = src_proj.transpose(0, 1)    # (seq_len, batch_size, d_model)
        src_proj = self.pos_encoder(src_proj)
        src_proj = src_proj.transpose(0, 1)    # (batch_size, seq_len, d_model)
        
        # 目标序列处理
        if tgt is None:
            # 推理时，使用零初始化目标序列
            tgt = torch.zeros(batch_size, self.pred_len, self.input_size).to(src.device)
        
        tgt_proj = self.input_projection(tgt)  # (batch_size, pred_len, d_model)
        tgt_proj = tgt_proj.transpose(0, 1)    # (pred_len, batch_size, d_model)
        tgt_proj = self.pos_encoder(tgt_proj)
        tgt_proj = tgt_proj.transpose(0, 1)    # (batch_size, pred_len, d_model)
        
        # 生成注意力掩码
        tgt_mask = self.transformer.generate_square_subsequent_mask(self.pred_len).to(src.device)
        
        # Transformer编码器-解码器
        output = self.transformer(src_proj, tgt_proj, tgt_mask=tgt_mask)  # (batch_size, pred_len, d_model)
        
        # 输出层
        output = self.output_layer(output)  # (batch_size, pred_len, output_size)
        
        return output


# 简化的Transformer模型（用于快速实验）
class SimpleTransformerModel(nn.Module):
    """简化版Transformer模型"""
    
    def __init__(self, input_size=8, d_model=32, nhead=4, num_layers=2, seq_len=20):
        super(SimpleTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer编码
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer_encoder(x, mask=mask)
        
        # 输出
        x = x.reshape(batch_size, -1)
        output = self.output_layer(x)
        
        return output.squeeze(-1)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
