import pandas as pd
from tqdm import tqdm
import json, gzip, re, pickle
pathIn = 'reddit-tomt-submissions.jsonl.gz'

with gzip.open(pathIn, 'rt') as f:
    d_all = []
    # Iterate through the lines in the file
    for line in tqdm(f):
        # Parse the line as a JSON object
        obj = json.loads(line)
        d_all.append(obj)
df_orig = pd.DataFrame(d_all)

df_solved = df_orig.loc[(df_orig['link_flair_text'] == 'Solved') | (df_orig['link_flair_text'] == 'Solved!')]

# create df_gold which contains all the questions for the Gold Answers that could be extracted
df_gold = df_solved.loc[df_solved['solved_utc'] != '']

def extract_category(title):
    ret = re.findall(r'\[.*?\]', title)
    return [x.lower() for x in ret]

movie_titles = {}
book_titles = {}
game_titles = {}
music_titles = {}

movie_queries = {}
book_queries = {}
game_queries = {}
music_queries = {}

movie_qrels = {}
book_qrels = {}
game_qrels = {}
music_qrels = {}

for i in tqdm(range(len(df_gold))):
        
    line = df_gold.iloc[i]    
    cat = extract_category(line['title'])
    
    if '[song]' in cat or '[music]' in cat:
        music_titles[line['id']] = line['title']
        music_queries[line['id']] = line['content']
        music_qrels[line['id']] = line['chosen_answer']

    if '[movie]' in cat: # or '[video]' in cat:
        movie_titles[line['id']] = line['title']
        movie_queries[line['id']] = line['content']
        movie_qrels[line['id']] = line['chosen_answer']


    if '[book]' in cat:
        book_titles[line['id']] = line['title']
        book_queries[line['id']] = line['content']
        book_qrels[line['id']] = line['chosen_answer']

    if '[game]' in cat:
        game_titles[line['id']] = line['title']
        game_queries[line['id']] = line['content']
        game_qrels[line['id']] = line['chosen_answer']


pickle.dump(movie_titles, open("movie_titles.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(book_titles, open("book_titles.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(game_titles, open("game_titles.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(music_titles, open("music_titles.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(movie_queries, open("movie_queries.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(book_queries, open("book_queries.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(game_queries, open("game_queries.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(music_queries, open("music_queries.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(movie_qrels, open("movie_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(book_qrels, open("book_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(game_qrels, open("game_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(music_qrels, open("music_qrels.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
