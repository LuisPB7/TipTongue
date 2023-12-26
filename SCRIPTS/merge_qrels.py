import pickle

movie_qrels = {}
book_qrels = {}
game_qrels = {}
music_qrels = {}

movie_parts = list(range(1,10))
music_parts = list(range(1,10))
game_parts = [1,2]
book_parts = [1,2,3]


parts = {'movie':movie_parts, 'music':music_parts, 'game':game_parts, 'book':book_parts}
qrels = {domain:{} for domain in parts}
domains = list(parts)

for domain in domains:
    for part in parts[domain]:

        titles = pickle.load(open("{}.qrels.part{}.pkl".format(domain, part), 'rb'))
        qrels[domain] = { **qrels[domain], **titles }


    pickle.dump(qrels[domain], open("{}.qrels.pkl".format(domain),'wb'), protocol=pickle.HIGHEST_PROTOCOL)

