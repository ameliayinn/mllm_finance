import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from math import pi

class SinCosEmbedding(nn.Module):
    """扩散步骤的频率自适应编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.logspace(0, 4, half_dim, base=10.0) * (4 / (half_dim - 1))
        self.register_buffer('freqs', freqs * (2 * pi))
        
    def forward(self, t):
        """t: [batch,] 扩散步数"""
        emb = t[:, None] * self.freqs[None, :]  # [batch, dim//2]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [batch, dim]

class AdaLN(nn.Module):
    """自适应层归一化（扩散步条件注入）"""
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        
    def forward(self, x, step_emb):
        # step_emb: [batch, d_model]
        gamma = self.gamma(step_emb) + 1  # 初始化为单位变换
        beta = self.beta(step_emb)
        return gamma[:, None, :] * self.ln(x) + beta[:, None, :]

class MarkovConstraint(nn.Module):
    """横截性条件约束模块"""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, h):
        # h: [batch, seq_len, d_model]
        return self.tanh(self.proj(h)) * 0.1  # 约束输出在[-0.1, 0.1]

class TextTimeSeriesDiffusion(nn.Module):
    def __init__(self, 
                 cond_len=50, 
                 pred_len=10, 
                 d_model=128,
                 nhead=4, 
                 num_layers=2,
                 dropout=0.1,
                 bert_name="bert-base-uncased",
                 freeze_bert=True):
        super().__init__()
        
        # 文本编码器
        self.bert = AutoModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.text_proj = nn.Linear(self.bert.config.hidden_size, d_model)
        
        # 时序编码器
        self.ts_encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model)
        
        # 扩散步编码
        self.step_encoder = nn.Sequential(
            SinCosEmbedding(d_model//2),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model))
        
        # 多模态融合
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)])
        
        # 去噪网络
        self.denoise_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                      dropout=dropout, batch_first=True)
            for _ in range(num_layers)])
        
        # 自适应归一化
        self.ada_ln = nn.ModuleList([AdaLN(d_model) for _ in range(num_layers)])
        
        # 事件门控
        self.event_gate = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Sigmoid())
        
        # 约束模块
        self.markov_constraint = MarkovConstraint(d_model)
        
        # 输出层
        self.output_proj = nn.Linear(d_model, 1)
        
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, context, noisy_target, diffusion_step, text_input):
        """
        Args:
            context: [batch, cond_len] 历史时序
            noisy_target: [batch, pred_len] 带噪预测目标
            diffusion_step: [batch,] 扩散步数
            text_input: 文本输入 (BERT格式)
        Returns:
            denoised: [batch, pred_len] 去噪结果
            latent: [batch, pred_len, d_model] 潜表示
            constraint: [batch, pred_len] 约束项
        """
        # 1. 编码输入特征
        # 文本特征 [batch, seq_len, d_model]
        text_feat = self.text_proj(self.bert(**text_input).last_hidden_state)
        
        # 时序特征 [batch, cond_len+pred_len, d_model]
        ts_input = torch.cat([context.unsqueeze(-1), noisy_target.unsqueeze(-1)], dim=1)
        ts_feat = self.ts_encoder(ts_input)
        
        # 扩散步特征 [batch, d_model]
        step_feat = self.step_encoder(diffusion_step)
        
        # 2. 跨模态注意力
        key_value = torch.cat([ts_feat, text_feat], dim=1)
        query = ts_feat[:, -self.pred_len:, :]  # 仅对预测部分做融合
        
        for attn in self.cross_attn:
            query, _ = attn(query, key_value, key_value)
        
        # 3. 结构化去噪
        h = query
        for layer, ln in zip(self.denoise_layers, self.ada_ln):
            # 事件门控
            gate = self.event_gate(torch.cat([h, text_feat.mean(dim=1, keepdim=True)], dim=-1))
            h = gate * h
            
            # 扩散步条件注入
            h = ln(h, step_feat)
            
            # 自注意力变换
            h = layer(h)
        
        # 4. 马尔可夫约束
        constraint = self.markov_constraint(h)
        h = h + constraint
        
        # 5. 输出处理
        denoised = self.output_proj(h).squeeze(-1)
        return denoised, h, constraint

    def compute_loss(self, y0, yk, denoised, latent, constraint, lambda_cons=0.1):
        """双目标损失计算"""
        # 重构损失
        recon_loss = F.mse_loss(denoised, y0)
        
        # 潜空间一致性损失
        latent_loss = F.mse_loss(self.ts_encoder(y0.unsqueeze(-1)), latent)
        
        # 约束惩罚项
        cons_loss = constraint.pow(2).mean()
        
        return recon_loss + lambda_cons * latent_loss + 0.01 * cons_loss