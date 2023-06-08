from typing import Optional
import torch
import torch.nn as nn
from module.models.modeling_bart import BartLearnedPositionalEmbedding
from copy import deepcopy

def expand_positional_encoding(embed: BartLearnedPositionalEmbedding, length, copy=True):
    """
    The embedding will be changed to length.
    copy: whether to return a new embedding. If copy is False, the input ```embed```
    will be changed.
    """
    if copy:
        embed = deepcopy(embed)
    if isinstance(embed, BartLearnedPositionalEmbedding):
        origin_num = embed.num_embeddings
        embed.num_embeddings = length + 2
    elif isinstance(embed, nn.Embedding):
        origin_num = embed.num_embeddings
        embed.num_embeddings = length
    else:
        RuntimeError(f'Embedding {type(embed)} not recognized')

    weight = embed.weight.clone().detach()
    if embed.num_embeddings < origin_num:
        new_weight = weight[:embed.num_embeddings]
    else:
        embed.weight = nn.Parameter(torch.Tensor(embed.num_embeddings, embed.embedding_dim))
        embed.reset_parameters()
        new_weight = embed.weight.clone().detach()
        new_weight[:origin_num] = weight
    embed.weight = nn.Parameter(new_weight)
    return embed

class TextEmbedding(nn.Module):

    def __init__(self, token_embed, position, embed_scale=1):
        """```embed_scale``` is for the compatibility to embedding in bart"""
        super(TextEmbedding, self).__init__()
        self.token_embed = token_embed
        self.pe = position
        self.embed_scale = embed_scale

    def forward(self, x):
        embed = self.token_embed(x) * self.embed_scale
        embed += self.pe(x.shape)
        return embed

class PositionalEncoding(nn.Embedding):
    """For compatibility of BartLearnedPositionalEmbedding"""

    def __init__(self, num_embeddings: int, embedding_dim: int, 
            padding_idx: Optional[int] = None, 
            max_norm: Optional[float] = None, 
            norm_type: float = 2, 
            scale_grad_by_freq: bool = False, 
            sparse: bool = False, 
            _weight: Optional[torch.Tensor] = None):
        super(PositionalEncoding, self).__init__(num_embeddings, 
                embedding_dim, padding_idx, max_norm, norm_type, 
                scale_grad_by_freq, sparse, _weight)
        self.reset_parameters()

    def forward(self, shape):
        bs, seq_len = shape[:2]
        positions = torch.arange(seq_len).expand((bs, seq_len)).to(self.weight.device)
        pe = super().forward(positions)
        return pe