import pickle, random

random.seed(0)

movie_queries = pickle.load(open("../DATA/movie_queries.pkl", 'rb'))
movie_qrels = pickle.load(open("../DATA/movie_qrels.pkl", 'rb'))
movie_titles = pickle.load(open("../DATA/movie_titles.pkl", 'rb'))

book_queries = pickle.load(open("../DATA/book_queries.pkl", 'rb'))
book_qrels = pickle.load(open("../DATA/book_qrels.pkl", 'rb'))
book_titles = pickle.load(open("../DATA/book_titles.pkl", 'rb'))

game_queries = pickle.load(open("../DATA/game_queries.pkl", 'rb'))
game_qrels = pickle.load(open("../DATA/game_qrels.pkl", 'rb'))
game_titles = pickle.load(open("../DATA/game_titles.pkl", 'rb'))

music_queries = pickle.load(open("../DATA/music_queries.pkl", 'rb'))
music_qrels = pickle.load(open("../DATA/music_qrels.pkl", 'rb'))
music_titles = pickle.load(open("../DATA/music_titles.pkl", 'rb'))


movie_qids = list(movie_qrels)
book_qids = list(book_qrels)
game_qids = list(game_qrels)
music_qids = list(music_qrels)

print(len(movie_qids))

movie_val_qids = random.sample(movie_qids, 100)
book_val_qids = random.sample(book_qids, 100)
game_val_qids = random.sample(game_qids, 100)
music_val_qids = random.sample(music_qids, 100)

movie_qids = [qid for qid in movie_qids if qid not in movie_val_qids]
book_qids = [qid for qid in book_qids if qid not in book_val_qids]
game_qids = [qid for qid in game_qids if qid not in game_val_qids]
music_qids = [qid for qid in music_qids if qid not in music_val_qids]

print(len(movie_qids))

movie_test_qids = random.sample(movie_qids, 100)
book_test_qids = random.sample(book_qids, 100)
game_test_qids = random.sample(game_qids, 100)
music_test_qids = random.sample(music_qids, 100)

movie_qids = [qid for qid in movie_qids if qid not in movie_test_qids]
book_qids = [qid for qid in book_qids if qid not in book_test_qids]
game_qids = [qid for qid in game_qids if qid not in game_test_qids]
music_qids = [qid for qid in music_qids if qid not in music_test_qids]


print(len(movie_qids))

train_movie_qrels = {qid:movie_qrels[qid] for qid in movie_qids}
train_book_qrels = {qid:book_qrels[qid] for qid in book_qids}
train_game_qrels = {qid:game_qrels[qid] for qid in game_qids}
train_music_qrels = {qid:music_qrels[qid] for qid in music_qids}

val_movie_qrels = {qid:movie_qrels[qid] for qid in movie_val_qids}
val_book_qrels = {qid:book_qrels[qid] for qid in book_val_qids}
val_game_qrels = {qid:game_qrels[qid] for qid in game_val_qids}
val_music_qrels = {qid:music_qrels[qid] for qid in music_val_qids}

test_movie_qrels = {qid:movie_qrels[qid] for qid in movie_test_qids}
test_book_qrels = {qid:book_qrels[qid] for qid in book_test_qids}
test_game_qrels = {qid:game_qrels[qid] for qid in game_test_qids}
test_music_qrels = {qid:music_qrels[qid] for qid in music_test_qids}


pickle.dump(train_movie_qrels, open("../DATA/train_movie_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(train_book_qrels, open("../DATA/train_book_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(train_game_qrels, open("../DATA/train_game_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(train_music_qrels, open("../DATA/train_music_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(val_movie_qrels, open("../DATA/val_movie_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_book_qrels, open("../DATA/val_book_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_game_qrels, open("../DATA/val_game_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_music_qrels, open("../DATA/val_music_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(test_movie_qrels, open("../DATA/test_movie_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(test_book_qrels, open("../DATA/test_book_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(test_game_qrels, open("../DATA/test_game_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(test_music_qrels, open("../DATA/test_music_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

