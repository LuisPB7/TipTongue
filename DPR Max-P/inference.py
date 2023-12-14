from DPR import KALE
from dataset import TestDataset
import torch, pickle, argparse
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix, vstack
import os
os.environ['CURL_CA_BUNDLE'] = ''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student', default='distilbert-base-uncased', type=str,
                        help='student to use (string from sentence-transformers)')
    parser.add_argument('--vocab_size', default=98304, type=int, help='vocabulary size')
    parser.add_argument('--k_queries', default=32, type=int, help='number of query expansion terms')
    parser.add_argument('--k_passages', default=256, type=int, help='number of passage expansion terms')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size per GPU')
    args = parser.parse_args()
    return args

args = parse_args()

passages = pickle.load(open("../../MSMARCO-PASSAGE/data/passages.pkl", 'rb'))
queries = pickle.load(open("../../MSMARCO-PASSAGE/data/queries.pkl", 'rb'))
dev_qids = pickle.load(open("../../MSMARCO-PASSAGE/data/qrels.dev.small.pkl", 'rb'))
dev_queries = {qid:queries[qid] for qid in dev_qids}

# Compute passages sparse representations #
dataset = TestDataset(passages, args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
sparse_representations = []
model = KALE(args).cuda().eval()
state_dict = torch.load("autoencoder.pt")
new_state_dict = {}

for key in state_dict:
    new_state_dict[key.replace("module.", "")] = state_dict[key]

model.load_state_dict(new_state_dict) #.cuda().eval()

#dense_nonzero = []
#dense_l1s = []
#sparse_l1s = []

passage_representations = []

with torch.no_grad():
    for data in dataloader:
        dense_inputs = data['passages'].cuda()
        atts = data['passages_attention'].cuda()
        projected  = model(dense_inputs, atts)
        passage_representations.append(projected.cpu())
        #dense_nonzero.append(dense_dims.cpu())
        #dense_l1s.append(dense_l1.cpu())
        #sparse = model.topK(projected)
        #sparse_l1 = sparse.sum(dim=-1) 
        #sparse_l1s.append(sparse_l1.cpu())
        #sparse_representations.append(csr_matrix(sparse.cpu()))

#pickle.dump(dense_nonzero, open("passages_dense_nonzero.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#pickle.dump(dense_l1s, open("dense_l1s.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#pickle.dump(sparse_l1s, open("sparse_l1s.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

passage_representations = torch.vstack(passage_representations)
pickle.dump(passage_representations, open("passage_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Compute queries sparse representations #
dataset = TestDataset(dev_queries, args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
query_representations = []

with torch.no_grad():
    for data in dataloader:
        dense_inputs = data['passages'].cuda()
        atts = data['passages_attention'].cuda()
        projected = model(dense_inputs, atts)
        query_representations.append(projected.cpu())
        #dense_nonzero.append(dense_dims.cpu())
        #sparse = model.topK(projected, is_query=True)
        #sparse_representations.append(csr_matrix(sparse.cpu()))

query_representations = torch.vstack(query_representations)
pickle.dump(query_representations, open("query_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#pickle.dump(dense_nonzero, open("queries_dense_nonzero.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


