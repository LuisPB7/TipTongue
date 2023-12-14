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

    def training_step(self, data, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        query = data['queries']
        query_attentions = data["queries_attention"]
        pos_passages = data['pos_passages']
        pos_passages_attentions = data['pos_passages_attention']

        # Process queries #
        cls_query = self(query, query_attentions)
        cls_query_dist = SyncFunction.apply(cls_query)

        # Process relevant document #
        cls_pos_passages = self(pos_passages, pos_passages_attentions)
        cls_pos_passages_dist = SyncFunction.apply(cls_pos_passages)

        # Compute loss #
        loss = self.rank_loss(cls_query_dist, cls_pos_passages_dist)

        self.training_step_outputs.append(loss.clone().detach().cpu())
        self.log("loss", loss)
        return loss

