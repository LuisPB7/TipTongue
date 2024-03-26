from model import DPR
import pickle, torch, argparse
from dataset import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='sentence-transformers/msmarco-bert-co-condensor', type=str,
                        help='Huggingface backbone to start training from')
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    parser.add_argument('--query_max_seq_len', default=512, type=int,
                        help='max seq length for query transformer inputs')
    parser.add_argument('--doc_max_seq_len', default=512, type=int, help='max seq length for doc transformer inputs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    args = parser.parse_args()
    return args


args = parse_args()

passages = pickle.load(open("../WIKIPEDIA/wikipedia_passages.pkl", 'rb'))
test_qids = pickle.load(open("../DATA/test_{}_qrels_human.pkl".format(args.domain), 'rb'))
queries = pickle.load(open("../DATA/{}_queries.pkl".format(args.domain), 'rb'))
test_queries = {qid: queries[qid] for qid in test_qids}

# Compute passages sparse representations #
dataset = TestDataset(passages, args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
representations = []
model = DPR(args).cuda().eval()
model.load_state_dict(torch.load("dpr.pt", map_location='cuda'))

with torch.no_grad():
    for data in tqdm(dataloader):
        dense_inputs = data['text'].cuda()
        atts = data['text_attention'].cuda()
        cls = model(dense_inputs, atts)
        representations.append(cls.cpu())

representations = torch.vstack(representations)
pickle.dump(representations, open("passage_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Compute queries sparse representations #
dataset = TestDataset(test_queries, args, is_query=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
representations = []

with torch.no_grad():
    for data in dataloader:
        dense_inputs = data['text'].cuda()
        atts = data['text_attention'].cuda()
        cls = model(dense_inputs, atts)
        representations.append(cls.cpu())

representations = torch.vstack(representations)
pickle.dump(representations, open("query_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
