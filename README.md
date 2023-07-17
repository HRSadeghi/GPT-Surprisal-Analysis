# GPT Surprisal Analysis

The surprisal criterion has been investigated on n-gram statistical language models before, but less attention has been paid to the investigation of this criterion on deep language models such as GPT. In this repository, this criterion is calculated on the GPT model and for a Persian dataset, for each of its words or sub-words.



Since the GPT tokenizer may break words into sub-words, we have performed surprisal calculation in two ways. In the first method, the surprisal value is calculated for each sub-word. But in the second method, the surprisal value of each word is calculated by combining the surprisals of its sub-words
