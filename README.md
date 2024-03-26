# Tip-of-the-Tongue

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

- DATA (contains ToT queries, query titles, qrels): https://mega.nz/file/QpBCxRZA#9Z6Q3AH2kB2c3jqMfjUx2SGcS3u12ZEc1wBvkpSLsMs
- WIKIPEDIA (contains Wikipedia documents, Wiki titles): https://mega.nz/file/cw4FzIxY#HeGKOXEzbgYP80_R3elA5nu5rinHj0p1OTXZ4ATN1oA
- TREC (Specific TREC ToT queries and Wikipedia split): https://mega.nz/file/Ytg1nTDC#cZ5_2Keys_bNMwXq_UY00TZMYzOJPca6MP7C79hNMSY




