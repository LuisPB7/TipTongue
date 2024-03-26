from dataset import *
from model import DPR
import argparse
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='sentence-transformers/msmarco-bert-co-condensor', type=str,
                        help='Huggingface backbone to start training from')
parser.add_argument('--domains', default='all', type=str,
                        help='Domain to train on')
parser.add_argument('--query_max_seq_len', default=512, type=int, help='max seq length for query transformer inputs')
parser.add_argument('--doc_max_seq_len', default=512, type=int, help='max seq length for doc transformer inputs')
parser.add_argument('--lr', default=2e-5, type=float, help='max learning rate')
parser.add_argument('--warmup_steps', default=0.1, type=float, help='portion warmup steps')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
args = parser.parse_args()

# Load datasets #
dataset = TrainDataset(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
# ------------- #

# Load model #
seed_everything(0, workers=True)
model = DPR(args)

trainer = pl.Trainer(max_epochs=args.epochs, precision="16-mixed", enable_progress_bar=True, accelerator="gpu", devices=[0], deterministic='warn')
trainer.fit(model=model, train_dataloaders=dataloader)

# ---------------------------------------------------------------- #
