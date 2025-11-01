Project Title
Simple Sentence-Level Spelling Autocorrector

Description
This is a basic Natural Language Processing (NLP) project that implements a custom spell checker for English sentences in Python. It identifies and corrects common spelling mistakes by comparing input words with a vocabulary derived from the English Word Frequency dataset.

The correction logic is built from scratch using edit-distance operations (deletion, insertion, swapping, and replacement). It selects the best candidate based on word frequency, without using any pretrained models or external APIs.

Features
Word-level spelling correction for simple sentences

Custom implementation using:

Edit distance 1

Frequency-based word probability

No pretrained models used

Designed to run in a Jupyter Notebook or Python terminal

Limitations
Grammar and punctuation correction is not included

Does not support full document files (e.g., PDF or Word)

No GUI or web interface

Only corrects words that are one or two edits away from known words

Dataset Used
English Word Frequency List from Kaggle
https://www.kaggle.com/datasets/rtatman/english-word-frequency