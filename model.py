import torch
import torch.nn as nn
from transformers import AutoModel

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        residual = query
        out, _ = self.mha(query, key, value)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class TextTimeSeriesDiffusion(nn.Module):
    def __init__(self, cond_len=50, pred_len=10, d_model=128, nhead=4, num_layers=2, dropout=0.1, dim_feedforward=512, bert_name="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.bert = AutoModel.from_pretrained(bert_name)
        d_bert = self.bert.config.hidden_size
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_proj = nn.Linear(d_bert, d_model)

        self.ts_embed = nn.Linear(1, d_model)
        self.n_layers = num_layers
        self.cross_attn_blocks = nn.ModuleList([MultiHeadAttnBlock(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.ts_ffn_blocks = nn.ModuleList([FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)])
        self.self_attn_blocks = nn.ModuleList([MultiHeadAttnBlock(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.self_ffn_blocks = nn.ModuleList([FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)])

        self.pred_queries = nn.Parameter(torch.randn(pred_len, d_model))
        self.out_linear = nn.Linear(d_model, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model))

    def encode_text(self, text_input):
        out_bert = self.bert(input_ids=text_input["input_ids"], attention_mask=text_input.get("attention_mask", None))
        hidden_states = out_bert.last_hidden_state
        text_emb = self.bert_proj(hidden_states)
        return text_emb

    def forward(self, context, x_t, t, text_input):
        B = context.size(0)
        text_emb = self.encode_text(text_input)
        len_text = text_emb.size(1)

        ts_cat = torch.cat([context, x_t], dim=-1)
        ts_cat = ts_cat.unsqueeze(-1)
        ts_emb = self.ts_embed(ts_cat)

        pred_q = self.pred_queries.unsqueeze(0).expand(B, -1, -1)

        combined_ts = torch.cat([ts_emb, pred_q], dim=1)
        total_len = combined_ts.size(1)

        pe_text = self.pos_embed[:, :len_text, :]
        text_emb = text_emb + pe_text

        pe_ts = self.pos_embed[:, len_text: len_text + total_len, :]
        combined_ts = combined_ts + pe_ts

        out_ts = combined_ts
        for i in range(self.n_layers):
            out_ts = self.cross_attn_blocks[i](out_ts, text_emb, text_emb)
            out_ts = self.ts_ffn_blocks[i](out_ts)
            out_ts = self.self_attn_blocks[i](out_ts, out_ts, out_ts)
            out_ts = self.self_ffn_blocks[i](out_ts)

        out_pred_q = out_ts[:, -self.pred_len:, :]
        logits = self.out_linear(out_pred_q).squeeze(-1)
        return logits
