import pickle
import time
import numpy as np
import faiss
from sklearn import preprocessing

passages = pickle.load(open("../../MSMARCO-PASSAGE/data/passages.pkl", 'rb'))
queries = pickle.load(open("../../MSMARCO-PASSAGE/data/qrels.dev.small.pkl", 'rb'))

N = 1000
pids = list(passages)

results = {}
scores = {}

embeddings = pickle.load(open("passage_embeddings.pkl", 'rb')).numpy().astype(np.float32)
question_embeddings = pickle.load(open("query_embeddings.pkl", 'rb')).numpy().astype(np.float32)

#embeddings = np.array( preprocessing.normalize(embeddings, norm='l2'), dtype="float32" )
#question_embeddings = np.array( preprocessing.normalize(question_embeddings, norm='l2'), dtype="float32" )

d = 1024

index = faiss.IndexFlatIP(d)

index.train(embeddings)
index.add(embeddings)

start = time.time()
D, I = index.search(question_embeddings, N)
end = time.time()

print("Retrieval took " + str(end-start) + " seconds")

for i, qid in enumerate(list(queries)):
    ix = list(I[i])
    results[qid] = [pids[j] for j in ix]
    scores[qid] = list(D[i])

pickle.dump(results, open("top1000.dev.ranks.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(scores, open("top1000.dev.scores.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
