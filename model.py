'''
Transformer model adapted from https://github.com/karpathy/nanoGPT with the following license:

MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def positionalencoding1d(d_model, length, is_cuda):
    #from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    device = 'cuda' if is_cuda else 'cpu'
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.to(device)


class SA_One_Head(nn.Module):
  def __init__(self, head_size, embed_size):
    super().__init__()
    self.head_size = head_size
    self.embed_size = embed_size
    self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
    self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
    self.value = nn.Linear(self.embed_size, self.head_size, bias=False)


  def forward(self, x):
    B, T, C = x.size()

    key = self.key(x)
    query = self.query(x)
    value = self.value(x)
    


    wei = query @ key.transpose(1,2)*(C**(-0.5))
    wei = F.softmax(wei, dim=2)

    out = wei @ value 

    return out
  
class SA_Multi_Head(nn.Module):
  def __init__(self, head_size, embed_size):
    super().__init__()
    self.head_size = head_size
    self.embed_size = embed_size
    self.num_heads = self.embed_size // self.head_size
    self.sa_heads = nn.ModuleList([SA_One_Head(self.head_size, self.embed_size) for i in range(self.num_heads)])

  def forward(self, x):
    x = torch.cat([sa(x) for sa in self.sa_heads], dim=-1)
    return x
  
class FeedForward(nn.Module):
  def __init__(self, embed_size):
    super().__init__()
    self.embed_size = embed_size
    self.f1 = nn.Linear(self.embed_size, self.embed_size*4)
    self.f2 = nn.Linear(self.embed_size*4, self.embed_size)
    
  def forward(self, x):
    x = self.f1(x)
    x = F.relu(x)
    x = self.f2(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self,embed_size, head_size):
    super().__init__()
    self.embed_size = embed_size
    self.head_size = head_size
    self.msa1 = SA_Multi_Head(self.head_size, self.embed_size)
    self.msa2 = SA_Multi_Head(self.head_size, self.embed_size)
    self.ff = FeedForward(self.embed_size)
    self.ln1 = nn.LayerNorm(self.embed_size)
    self.ln2 = nn.LayerNorm(self.embed_size)
    self.ln3 = nn.LayerNorm(self.embed_size)

  def forward(self,x):
    x = x + self.msa1(self.ln1(x))
    x = x + self.msa2(self.ln2(x))
    x = x + self.ff(self.ln3(x))
    return x
  
class SentimentAnalysisModel(nn.Module):
  def __init__(self, vocab_size, embed_size, head_size, num_layers, num_classes):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.head_size = head_size
    self.num_classes = num_classes
    self.emb = nn.Embedding(self.vocab_size, self.embed_size)
    self.tr_blocks = nn.Sequential(*[TransformerBlock(self.embed_size, self.head_size) for i in range(num_layers)])
    self.l = nn.Linear(self.embed_size, self.num_classes)

  def forward(self, x, y=None):
    x = self.emb(x)
    B, T, C = x.size()
    pos_embeddings = positionalencoding1d(C, T, next(self.parameters()).is_cuda)
    x = x + pos_embeddings
    x = self.tr_blocks(x)
    x = self.l(x)
    B, T, C = x.size()
    prediction = x[:, -1, :]
    if y is None:
      loss = None
    else:
      loss = F.cross_entropy(prediction,y)

    return prediction, loss

