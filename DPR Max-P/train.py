import torch
import argparse
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from DPR import DPR
from loss import ContrastiveLoss
from utils import cos_sim
from dataset import TrainDataset
from transformers import get_scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MASTER_SEED = 0  # Your master seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='OpenMatch/co-condenser-large-msmarco', type=str,
                        help='Huggingface backbone to start training from')
    parser.add_argument('--query_max_seq_len', default=128, type=int, help='max seq length for query transformer inputs')
    parser.add_argument('--doc_max_seq_len', default=512, type=int, help='max seq length for doc transformer inputs')
    parser.add_argument('--lr', default=2e-5, type=float, help='max learning rate')
    parser.add_argument('--warmup_steps', default=0.1, type=float, help='portion warmup steps')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    args = parser.parse_args()
    return args


def main(args):
    process_seed = MASTER_SEED
    set_seed(process_seed)

    # Define your model, dataset, dataloader, and criterion
    model = DPR(args).cuda()

    dataset = TrainDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    scale = 1.0
    sim_f = cos_sim
    rank_loss = ContrastiveLoss(scale=scale, similarity_fct=sim_f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # Maximum LR
    scaler = GradScaler()

    steps_per_epoch = len(dataloader)
    total_steps = args.epochs * steps_per_epoch
    increase_lr_steps = args.warmup_steps * total_steps

    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=increase_lr_steps,
                              num_training_steps=total_steps)

    for epoch in range(args.epochs):

        start = time.time()
        loss_avg = 0.0
        n_samples = len(dataloader)
        for i, data in enumerate(dataloader):
            query = data['queries'].cuda()
            query_attentions = data["queries_attention"].cuda()
            pos_passages = data['pos_passages'].cuda()
            pos_passages_attentions = data['pos_passages_attention'].cuda()

            optimizer.zero_grad()

            with autocast():
                # Process queries #
                query_rep = model(query, query_attentions)

                # Process relevant document #
                pos_passage_rep = model(pos_passages, pos_passages_attentions)

                # Compute Distillation #
                loss = rank_loss(query_rep, pos_passage_rep)

                loss_avg = loss_avg + loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        end = time.time()
        epoch_minutes = (end - start) / 60
        loss_avg = loss_avg / n_samples

        last_lr = scheduler.get_last_lr()[0]
        print(
            f"FINISHED EPOCH {epoch}. LOSS {loss_avg} ;  LR = {last_lr} ;  TIME = {epoch_minutes} mins",
            flush=True)
        torch.save(model.state_dict(), "dpr.pt")

    torch.save(model.state_dict(), "dpr.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
