import pickle
import csv

#test-culture-ahrtsdlgra-con01a Q0 test-culture-ahrtsdlgra-con01a 1 319.341888 Anserini

queries = pickle.load(open("../../MSMARCO-PASSAGE/data/qrels.dev.small.pkl", 'rb'))
results = pickle.load(open("top1000.dev.reranked.pkl", 'rb'))
#scores = pickle.load(open("top1000.dev.scores.pkl", 'rb'))

with open('run.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter=' ')

    for qid in queries:
        ranks = results[qid]
        #scor = scores[qid]
        for i,pid in enumerate(ranks):
            
            tsv_writer.writerow([qid, 'Q0', pid, str(i+1), str(1000-i), 'Anserini'])

