import  pickle

import csv

queries = pickle.load(open("../data/dev_queries.pkl", 'rb'))
results = pickle.load(open("top1000.dev.ranks.pkl", 'rb'))
scores = pickle.load(open("top1000.dev.scores.pkl", 'rb'))

with open('run.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter=' ')

    for qid in queries:
        ranks = results[qid]
        scor = scores[qid]
        for i,pid in enumerate(ranks):
            
            tsv_writer.writerow([qid, 'Q0', pid, str(i+1), str(1000-i), 'Anserini'])

