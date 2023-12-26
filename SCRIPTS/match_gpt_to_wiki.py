import pickle, difflib

wikipedia_titles = pickle.load(open("../WIKIPEDIA/titles_to_ids.pkl", 'rb'))
documents = pickle.load(open("../WIKIPEDIA/wikipedia_documents.pkl", 'rb'))

movie_answers = pickle.load(open("movie_gpt_answers.pkl", 'rb'))
book_answers = pickle.load(open("book_gpt_answers.pkl", 'rb'))
game_answers = pickle.load(open("game_gpt_answers.pkl", 'rb'))
music_answers = pickle.load(open("music_gpt_answers.pkl", 'rb'))

title_list = list(wikipedia_titles)

def is_disambiguation(wiki_string):
    return "may refer to" in wiki_string or "may also refer to" in wiki_string

movie_suffixes = [' (movie)', ' (film)', ' (movie series)', ' (film series)', '']
book_suffixes = [ ' (book)', ' (novel)', ' (book series)', ' (novel series)', '']
game_suffixes = [ ' (video game)', ' (video game series)', ''  ]
music_suffixes = [ ' (song)', ''  ]

answers = {'movie': movie_answers, 'book':book_answers, 'game':game_answers, 'music':music_answers}
suffixes = {'movie': movie_suffixes, 'book':book_suffixes, 'game':game_suffixes, 'music':music_suffixes}

#BOOK: 20562
#MOVIE: 84106
#GAME: 13219
#MUSIC: 82116

domain = "music"
part = 9

domain_answers = answers[domain]
domain_suffixes = suffixes[domain]

qrels = {}
for i, qid in enumerate( list(domain_answers)[((part-1)*10000):(part*10000)] ):

    print(i)
    title = domain_answers[qid]
    matched = False

    for suff in domain_suffixes:

        title_suffix = title + suff

        try:
            did = wikipedia_titles[title_suffix]
            document = documents[did]
            if not is_disambiguation(document):
                qrels[qid] = did
                matched = True
                break
        except:
            pass


    if matched:
        continue

    # Now try a string similarity based approach #
    closest = difflib.get_close_matches(title, title_list, cutoff=0.9)
    if closest != []:
        print("TITLE: {}".format(title))
        print("CLOSEST: {}".format(closest[0]))
        print()
        did = wikipedia_titles[closest[0]]
        document = documents[did]
        if not is_disambiguation(document):
            qrels[qid] = did
        else:
            print("Disambiguation :(")


pickle.dump(qrels, open("{}.qrels.part{}.pkl".format(domain, part), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

