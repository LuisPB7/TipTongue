import pickle
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    args = parser.parse_args()
    return args

args = parse_args()

queries = pickle.load(open("../DATA/test_{}_qrels_human.pkl".format(args.domain), 'rb'))
results = pickle.load(open("top1000.test.ranks.{}.pkl".format(args.domain), 'rb'))

with open('run.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter=' ')

    for qid in queries:
        ranks = results[qid]
        for i,pid in enumerate(ranks):
            
            tsv_writer.writerow([qid, 'Q0', pid, str(i+1), str(1000-i), 'Anserini'])

