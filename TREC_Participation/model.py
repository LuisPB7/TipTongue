from utils import *

# define the LightningModule
class DPR(pl.LightningModule):
    def __init__(self, ranking_loss):

        super().__init__()
        self.lm = AutoModel.from_pretrained("OpenMatch/co-condenser-large-msmarco")
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        warmup_steps = int( 0.1 * self.trainer.estimated_stepping_batches )
        print("Estimated stepping batchs is {}".format(self.trainer.estimated_stepping_batches), flush=True)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "loss"}
    
    # This gets called when an epoch finishes #
    def on_train_epoch_end(self):
        mean = torch.stack(self.training_step_outputs).mean()
        if self.global_rank == 0:
            self.training_step_outputs.clear()
            print("( Loss: {:.4f} )".format(mean), flush=True)
            torch.save(self.state_dict(), "dpr.pt")




