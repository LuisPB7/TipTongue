# Code and data for the SIGIR '24 paper "Generalizable Tip-of-the-Tongue Retrieval with LLM Re-ranking"

**First step: Collecting data**

1) Get the entirety of tip-of-the-tongue queries


```wget https://files.webis.de/corpora/corpora-webis/known-item-question-performance-prediction/reddit-tomt-submissions.jsonl.gz```

2) Filter queries by categories: movies, books, games, music. These queries are solved, with an answer in natural language (i.e., a Reddit reply).


```python3 SCRIPTS/extract_categories.py```

3) Have GPT-3.5 extract the item title from the natural language answer


```python3 SCRIPTS/gpt_titles.py```

4) Match the GPT-3.5 extract titles to Wikipedia titles, if they can be matched. This further narrows down the amount of queries.


```python3 SCRIPTS/match_gpt_to_wiki.py```

5) Now that we have all the available queries (i.e., the ones with an answer that maps to a Wiki title), we split the data in training/val/test.


```python3 SCRIPTS/split_dataset.py```

6) Optionally, since every query is mapped to only one document, we can create lists of samples, which are (query_text, document_text) tuples.


```python3 SCRIPTS/process_qrels.py```

This is the process for collecting the data. We recommend instead downloading the data already processed into these three folders. These files are already human filtered (i.e., the testing queries).

- DATA (contains ToT queries, query titles, qrels): https://mega.nz/file/BogxwCII#U7r9mNwNJXQSyJgvtCjgK-XbG5qrIUZsyF6BRpGlx2A
- WIKIPEDIA (contains Wikipedia documents, Wiki titles): https://mega.nz/file/cw4FzIxY#HeGKOXEzbgYP80_R3elA5nu5rinHj0p1OTXZ4ATN1oA
- TREC (Specific TREC ToT queries and Wikipedia split): https://mega.nz/file/Ytg1nTDC#cZ5_2Keys_bNMwXq_UY00TZMYzOJPca6MP7C79hNMSY

**Second step: Training DPR on the collected data**

```sh DPR/script.sh``` will train a DPR model with every domain, and evaluate on a certain domain. Do look at the command line arguments to change these options. The link to download the weights is inside the DPR folder.

**November 2023 TREC Participation**

The folder ```TREC_Participation``` contains the dataset, code, and DPR weights for the TREC participation, submitted in August of 2023 and presented in November of 2023. The 2023 TREC ToT task differs from the SIGIR paper task. For example, TREC only evaluates on the movie domain, with its own 150 TREC queries. The TREC DPR model was only trained on movie queries. The methodology to obtain the training dataset was the same, although there are some differences in the dataset. For TREC, there is no test split, since that was given by the organizers. Additionally, the process of keeping/discarding movie queries was less strict, resulting in more movie queries for training (around 84k). We recommend using the methods and data from the SIGIR article instead.


