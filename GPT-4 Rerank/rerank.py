import os
import ast
import openai
openai.api_key = "" # Fill
import pickle
from difflib import get_close_matches
import json

model_name = "gpt-4"

def prompt_model(prompt, model_name):
    completion = openai.ChatCompletion.create(
      model=model_name,
      messages=[
        {"role": "user", "content": prompt},
      ]
    )
    return completion.choices[0].message.content



queries = pickle.load(open("../data/dev_queries.pkl", 'rb'))
titles = pickle.load(open("../data/titles.pkl", 'rb'))
qrel = pickle.load(open("../data/dev_qrel.pkl", 'rb'))
ranks = pickle.load(open("top1000.dev.ranks.pkl", 'rb'))

answers = {}

K = 100

def convert_to_list(answer):

    movies = []

    split = answer.split("\n")

    for i,string in enumerate(split):
        if string.startswith("1"):
            true_movies = split[i:]
            break
    try:
        for movie in true_movies:
            mov = movie.split(".")
            mov = '.'.join(mov[1:])
            if mov[0] == ' ':
                mov = mov[1:]
            if mov[-1] == ' ':
                mov = mov[:-1]
            movies.append(mov)
    except:
        return []

    return movies


def rerank(query, movie_list):

    prompt = "I am going to give you a question and a list of movies. Re-order the movies according to the likelihood that the question refers to the movie.  " \
             "Format the answer as a numbered list of {} movies. Keep the movie names I give you. " \
             "Be direct and return the list ONLY. Stick with movies from the list ONLY.\n\nQUESTION: {}\n\nMOVIE LIST: {}".format(K, query, '\n'.join(movie_list))

    

    succeeded = False
    while not succeeded:
        try:
            answer = prompt_model(prompt, model_name)
            succeeded = True
        except:
            succeeded = False

    llm_list = convert_to_list(answer)

    if llm_list == []:
        print(answer)
        return []

    candidates = movie_list.copy()

    # Build re-ranked list #
    reranked_list = []
    for title in llm_list:

        if title in reranked_list:
            continue

        if title in candidates:
            reranked_list.append(title)
            candidates.remove(title)
        else:
            closest = get_close_matches(title, candidates, cutoff=0.0)[0]
            print("Title: {}".format(title))
            print("Closest: {}".format(closest))
            print("Candidates: {}".format(candidates))
            reranked_list.append(closest)
            candidates.remove(closest)

    # If re-ranked list does not contain K elements, need to add remaining by the same order #
    if len(candidates) > 0:
        reranked_list = reranked_list + candidates

    return reranked_list



for i, qid in enumerate(queries):

    print(i, flush=True)

    query = queries[qid]

    top1000 = ranks[qid][:K]

    movies = [titles[pid] for pid in top1000]

    answers[qid] = rerank(query, movies)


pickle.dump(answers, open("top{}.dev.reranked.pkl".format(K), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
