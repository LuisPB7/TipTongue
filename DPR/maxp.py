import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    args = parser.parse_args()
    return args


args = parse_args()

ranks = pickle.load(open("top10k.test.ranks.pkl", 'rb'))

new_ranks = {qid:[] for qid in ranks}

for qid in ranks:

    for pid in ranks[qid]:

        true_pid = pid.split("-")[0]
        if true_pid not in new_ranks[qid]:
            new_ranks[qid].append(true_pid)
        if len(new_ranks[qid]) == 1000:
            break


pickle.dump(new_ranks, open("top1000.test.ranks.{}.pkl".format(args.domain), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


