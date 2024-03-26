import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str, help='Domain to train on')
    args = parser.parse_args()
    return args

args = parse_args()

ranks = pickle.load(open("../top1000.test.ranks.{}.pkl".format(args.domain), 'rb'))

new_ranks = {qid:[[] for _ in range(10)] for qid in ranks}

for qid in ranks:

    bucket = 0
    for i,pid in enumerate(ranks[qid]):

        new_ranks[qid][bucket].append(pid)
        bucket = bucket +  1
        if bucket == 10:
            bucket = 0

merged_ranks = {qid:[] for qid in ranks}

for qid in new_ranks:

    for bucket in new_ranks[qid]:

        merged_ranks[qid] = merged_ranks[qid] + bucket


pickle.dump(merged_ranks, open("top1000.test.ranks.robin.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


