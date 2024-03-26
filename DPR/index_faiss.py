import pickle
import time
import numpy as np
import faiss, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    args = parser.parse_args()
    return args


args = parse_args()

queries = pickle.load(open("../DATA/test_{}_qrels_human.pkl".format(args.domain), 'rb'))
passages = pickle.load(open("../WIKIPEDIA/wikipedia_passages.pkl", 'rb'))

N = len(passages)
pids = list(passages)

results = {}

embeddings = pickle.load(open("passage_embeddings.pkl", 'rb')).numpy().astype(np.float32)
question_embeddings = pickle.load(open("query_embeddings.pkl", 'rb')).numpy().astype(np.float32)

d = 1024

index = faiss.IndexFlatIP(d)

index.train(embeddings)
index.add(embeddings)

start = time.time()
D, I = index.search(question_embeddings, 10000)
end = time.time()

print("Retrieval took " + str(end-start) + " seconds")

for i, qid in enumerate(list(queries)):
    ix = list(I[i])
    results[qid] = [pids[j] for j in ix]

pickle.dump(results, open("top10k.test.ranks.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
