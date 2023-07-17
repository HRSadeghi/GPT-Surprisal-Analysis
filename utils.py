import math
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn as F



def surprisal(p):
    if p == 0:
        S = math.inf
        return -S
    S = math.log(p, 2)
    return -S

def load_word_tokens(file_path):
    words_file = open(file_path ,"r")
    words = words_file.readlines()
    words_file.close()
    words = [w.strip() for w in words]
    return words


def load_dataset(file_path):
    file_ = open(file_path ,"r")
    file1 = file_.readlines()
    file_.close()
    file1 = [d.strip() for d in file1]
    return file1



def build_test_set_from_words(words, test_data):
    len__ = 0
    cleaned_test_data = []
    for i in range(len(test_data)):
        cleaned_test_data += [words[len__:len__ + len(test_data[i].split(' '))]]
        len__ += len(test_data[i].split(' '))
    return cleaned_test_data


def get_pretrained_Persian_GPT2():
    tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
    model = GPT2LMHeadModel.from_pretrained('bolbolzaban/gpt2-persian')

    return tokenizer, model



def calculate_prob_batch(model, tokenizer, test_set, return_subtokens = False):
    outputs = []
    for s in tqdm(test_set):
        soft = F.Softmax(dim=2)
        inputs = tokenizer(' '.join(s), return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        out = soft(model(input_ids, attention_mask=attention_mask)['logits'])
        ps = [float(out[0][i][x]) for i,x in enumerate(inputs['input_ids'][0][1:-1])]
        outputs.append(ps[:])
        del inputs, out
    return outputs


def get_subtokens(tokenizer, test_set):
    subtokens = []
    for s in test_set:
        inputs = tokenizer(' '.join(s), return_tensors="pt")
        subtokens += inputs['input_ids'][0][1:-1].tolist()
    return subtokens

def tokenizer_alignment(tokenizer, test_set):
    out = []
    for x in test_set:
        temp_list = []
        for y in x:
            temp = tokenizer(y, return_tensors="pt")['input_ids'][0][1:-1]
            temp_list.append(temp.tolist())
        out.append(temp_list)
    return [[len(y) for y in x] for x in out]



def calculate_sequential_surprisal(prob_seq):
    temp = [surprisal(prob_seq[i]) for i in range(len(prob_seq))]
    return temp


def align_with_real_word_tokens(seq, alignment, type = 'sum'):
    output = []
    if type == 'sum':
        count = 0
        for x in alignment:
            aln = seq[count: count + x]
            output += [np.average(aln)]
            count += x
        return output
    else:
        count = 0
        for x in alignment:
            aln = seq[count: count + x]
            output += [np.product(aln)**(1/x)]
            count += x
        return output


def final_eval(model, tokenizer, test_set):
    outputs = calculate_prob_batch(model, tokenizer, test_set)
    alignment = tokenizer_alignment(tokenizer, test_set)
    probs = [align_with_real_word_tokens(outputs[i], alignment[i], type = 'product') for i in range(len(outputs))]
    eval = []
    for i in range(len(outputs)):
        A = align_with_real_word_tokens(calculate_sequential_surprisal(outputs[i]), alignment[i])
        eval.append(A)
    return eval, probs


def final_eval2(model, tokenizer, test_set):
    probs = calculate_prob_batch(model, tokenizer, test_set)
    eval = []
    for i in range(len(probs)):
        A = calculate_sequential_surprisal(probs[i])
        eval.append(A)
    return eval, probs


def df_to_csv(test_set, probability, surprisal_list, file_path = './word_list_with_evaluation.csv'):
    final_out = []
    for i, s in enumerate(test_set):
        for j, x in enumerate(s):
            final_out.append({'word': x, 'probability': probability[i][j], 'surprisal': surprisal_list[i][j]})
    df = pd.DataFrame(final_out)
    df.to_csv(file_path, sep='\t', index=None)
    return df


def df_to_csv2(tokenizer, test_set, probability, surprisal_list, file_path = './word_list_with_evaluation.csv'):
    sub__ = get_subtokens(tokenizer, test_set)
    w__ = tokenizer.convert_ids_to_tokens(sub__)
    probability__ = [y for x in  probability for y in x]
    surprisal_list__ = [y for x in  surprisal_list for y in x]

    final_out = []
    for i,x in enumerate(w__):
        final_out.append({'word': x, 'probability': probability__[i], 'surprisal': surprisal_list__[i]})
    df = pd.DataFrame(final_out)
    df.to_csv(file_path, sep='\t', index=None)
    return df