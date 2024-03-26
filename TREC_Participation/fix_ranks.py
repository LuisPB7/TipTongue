import pickle

ranks = pickle.load(open("top1000.dev.ranks.pkl", 'rb'))
scores = pickle.load(open("top1000.dev.scores.pkl", 'rb'))

new_ranks = {qid:[] for qid in ranks}

for qid in ranks:

    for pid in ranks[qid]:

        true_pid = pid.split("-")[0]
        if true_pid not in new_ranks[qid]:
            new_ranks[qid].append(true_pid)
        if len(new_ranks[qid]) == 1000:
            break


pickle.dump(new_ranks, open("top1000.dev.ranks.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


