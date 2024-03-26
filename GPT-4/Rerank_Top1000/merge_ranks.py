import  pickle

ranks = pickle.load(open("top1000.test.reranked.robin.pkl", 'rb'))

new_ranks = {qid:[] for qid in ranks}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

for qid in ranks:

    pids = ranks[qid]

    new_rank = []

    batched_pids = list( batch(pids, 100) )

    not_empty=True
    while not_empty:

        for i, b in enumerate(batched_pids):

            new_rank.append(batched_pids[i][0])
            batched_pids[i] = batched_pids[i][1:]
            print(len(b))

        not_empty = False
        for b in batched_pids:
            if len(b) != 0:
                not_empty=True
                break

    new_ranks[qid] = new_rank

pickle.dump(new_ranks, open("top1000.test.reranked.robin.merged.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)




