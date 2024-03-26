import os
import torch
import itertools
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import argparse
import torch.nn.functional as F
from scipy.sparse import csr_matrix, vstack
import json
import csv
from lightning_fabric.utilities.seed import seed_everything
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from transformers import AutoTokenizer, AutoModel, get_scheduler, DistilBertForMaskedLM, AutoModelForMaskedLM
from lightning.pytorch.loggers import TensorBoardLogger

# Variables #
NUM_EPOCHS = 10
BATCH_SIZE = 16
MAX_SEQ_LEN = 512
# ------- #

