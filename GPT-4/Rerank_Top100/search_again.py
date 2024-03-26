import pickle
import os
import ast
import openai
import argparse
openai.api_key = "" # Your key here
from difflib import get_close_matches
import json

model_name = "gpt-4-1106-preview"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    args = parser.parse_args()
    return args


args = parse_args()

def prompt_model(prompt, model_name):
    completion = openai.ChatCompletion.create(
      model=model_name,
      messages=[
        {"role": "user", "content": prompt},
      ]
    )
    return completion.choices[0].message.content


K = 100

reranks = pickle.load(open("top{}.test.reranked.pkl".format(K), 'rb'))

lens = [len(reranks[qid]) for qid in reranks]

queries = pickle.load(open("../../DATA/{}_queries.pkl".format(args.domain), 'rb'))
titles = pickle.load(open("../../WIKIPEDIA/wikipedia_titles.pkl", 'rb'))
qrel = pickle.load(open("../../DATA/test_{}_qrels_human.pkl".format(args.domain), 'rb'))
ranks = pickle.load(open("../top1000.test.ranks.{}.pkl".format(args.domain), 'rb'))
queries = {qid:queries[qid] for qid in qrel}

def convert_to_list(answer):
    items = []

    split = answer.split("\n")

    for i, string in enumerate(split):
        if string.startswith("1"):
            true_items = split[i:]
            break
    try:
        for item in true_items:
            it = item.split(".")
            it = '.'.join(it[1:])
            if it[0] == ' ':
                it = it[1:]
            if it[-1] == ' ':
                it = it[:-1]
            items.append(it)
    except:
        return []

    return items


def rerank(query, item_list):
    prompt = "I am going to give you a question and a list of items. Re-order the items according to the likelihood that the question refers to the item.  " \
             "Format the answer as a numbered list of {} items. Keep the item names I give you. " \
             "Be direct and return the list ONLY. Stick with items from the list ONLY.\n\nQUESTION: {}\n\nITEM LIST: {}".format(
        K, query, '\n'.join(item_list))

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

    candidates = item_list.copy()

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


for i, qid in enumerate(reranks):

    if lens[i] == 0:
        print(i, flush=True)

        query = queries[qid]

        top1000 = ranks[qid][:K]

        items = [titles[pid] for pid in top1000]

        reranks[qid] = rerank(query, items)

"""
for i, qid in enumerate(qrel):

    if qid not in reranks:
        print(i, flush=True)

        query = queries[qid]

        top1000 = ranks[qid][:K]

        items = [titles[pid] for pid in top1000]

        reranks[qid] = rerank(query, items)
"""
pickle.dump(reranks, open("top{}.test.reranked.pkl".format(K), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
