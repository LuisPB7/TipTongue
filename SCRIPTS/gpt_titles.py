# SCRIPT THAT ASKS CHATGPT TO CONVERT MOVIE/BOOK/GAME/MUSIC NATURAL LANGUAGE ANSWERS INTO ITEM TITLES #

import pickle
import openai
openai.api_key = "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"
from difflib import get_close_matches

model_name = "gpt-3.5-turbo"

def prompt_model(prompt, model_name):
    completion = openai.ChatCompletion.create(
      model=model_name,
      messages=[
        {"role": "user", "content": prompt},
      ]
    )
    return completion.choices[0].message.content



movie_qrels = pickle.load(open("../DATA/movie_qrels.pkl", 'rb'))
book_qrels = pickle.load(open("../DATA/book_qrels.pkl", 'rb'))
game_qrels = pickle.load(open("../DATA/game_qrels.pkl", 'rb'))
music_qrels = pickle.load(open("../DATA/music_qrels.pkl", 'rb'))

domain = "music"

domain_qrels = {'movie': movie_qrels, 'book':book_qrels, 'game':game_qrels, 'music':music_qrels}

generic_prompt = "I am going to give you a textual description. That text contains the title of a {}. Extract the title for me. I am going to give you two examples of how you should answer, and then you should extract the title. Do not engage in platitudes. Do not bloviate. Do not ramble. Be direct and efficient in speech and intention. Do not include warnings or caveats about your limitations as an AI. Return the Answer only. \n\nEXAMPLES:{}\n\nTEST:{}"

context_examples = \
        {\
        'movie': "Text: Is the sound an approaching train? Might be [The Godfather](https://youtu.be/kSQqv2UuvC0)\nAnswer: The Godfather\n\nText: I think it's The Zombie Diaries\nAnswer: The Zombie Diaries"  ,\
        'book': "Text: Cosmic, by Frank Cottrell Boyce?\nAnswer: Cosmic\n\nText: [Pancakes for Findus](https://www.goodreads.com/book/show/2387509.Pancakes_for_Findus) by Sven Nordqvist?\nAnswer: Pancakes for Findus"  ,\
        'game': "Text: The Vanishing of Ethan Carter https://youtu.be/HINFL5YrXMA\nAnswer: The Vanishing of Ethan Carter\n\nText: Twisted Metal 2? It's for the playstation 1 though.\nAnswer: Twisted Metal 2"  ,\
        'music': "Text: [Semi-Charmed Life by Third Eye Blind?](http://www.youtube.com/watch?feature=player_detailpage&amp;v=MyjTrwOMSO4#t=12s)\nAnswer: Semi-Charmed Life\n\nText: With Every Heartbeat by Robyn Edit: Wow, this song has quite a few covers and remixes.\nAnswer: With Every Heartbeat"  \
         }

answers = {}

qrels = domain_qrels[domain]

for i,qid in enumerate(list(qrels)):

    print(i)

    test = "Text: {}\nAnswer: ".format(qrels[qid])
    try:
        answer = prompt_model( generic_prompt.format(domain, context_examples[domain], test)   , model_name)
    except:
        pickle.dump(answers, open("{}_gpt_answers_intermediate.pkl".format(domain), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    answers[qid] = answer


pickle.dump(answers, open("{}_gpt_answers.pkl".format(domain), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

