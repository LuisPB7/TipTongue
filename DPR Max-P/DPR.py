import torch
import torch.nn as nn
from transformers import AutoModel


# define the LightningModule
class DPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lm = AutoModel.from_pretrained(args.backbone)
        self.projection = nn.Linear(1024, 1024)
        self.tanh = nn.Tanh()
        self.rank_loss = ranking_loss
        self.training_step_outputs = []

    def encode(self, x, attentions):
        model_output = self.lm(x, attentions, return_dict=False)
        token_embeddings = model_output[0]

        # Perform pooling
        cls = self.cls_pooling(token_embeddings)

        return cls

    def mean_pooling(self, x, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, x):
        return x[:, 0]

    def forward(self, x, attentions):
        cls = self.encode(x, attentions)
        cls = self.projection(cls)
        cls = self.tanh(cls)
        return cls
