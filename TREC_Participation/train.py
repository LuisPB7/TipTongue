from utils import *
from loss import *
from dataset import *
from model import DPR
from imports import *


# Load datasets #
dataset = TrainDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
# ------------- #

# Define losses #
ranking_c = ContrastiveLoss
scale = 1.0
sim_f = dot_product
rank_loss = ranking_c(scale=scale, similarity_fct=sim_f)
# ------------ #


# Load model #
seed_everything(0, workers=True)

model = DPR(rank_loss)

ddp = DDPStrategy(find_unused_parameters=True, process_group_backend="gloo")

trainer = pl.Trainer(max_epochs=NUM_EPOCHS, precision=16, enable_progress_bar=True, accelerator="gpu", devices=[1,2,4], num_nodes=1, strategy=ddp, deterministic='warn')
print("FITTING...")
trainer.fit(model=model, train_dataloaders=dataloader)

# ---------------------------------------------------------------- #
