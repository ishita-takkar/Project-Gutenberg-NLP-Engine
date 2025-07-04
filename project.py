# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    ans = requests.get(url)
    ans.raise_for_status()
    text = ans.text.replace('\r\n', '\n')
    data = text.split('\n') 
    
    start_comment = [i for i, line in enumerate(data) if "*** START OF THE PROJECT GUTENBERG EBOOK" in line]
    end_comment = [i for i, line in enumerate(data) if "*** END OF THE PROJECT GUTENBERG EBOOK" in line]
    start_ind = start_comment[0]
    end_ind = end_comment[0]
    output = '\n'.join(data[start_ind + 1: end_ind])
    
    time.sleep(0.5)
    return '\n' + output

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string: str):
    raw_str = re.sub(r'\n{2,}', ' \x03 \x02 ', book_string.strip())
    raw_token = ['\x02'] + re.findall(r'\w+|[^\w\s]', raw_str) + ['\x03']
    return raw_token


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        series = pd.Series(tokens)
        elem = series.unique()
        prob = 1/len(elem)
        probs = pd.Series(prob, index = elem)
        return probs
    
    def probability(self, words):
        probability = 1.0
        for word in words:
            if word in self.mdl:
                probability *= self.mdl[word]
            else:
                probability = 0.0
        return float(probability)
        
    def sample(self, M):
        tokens = self.mdl.sample(n=M, weights=self.mdl.values, replace=True, random_state=1)
        randomized = ' '.join(tokens.index)
        return randomized


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        series = pd.Series(tokens)
        counts = series.value_counts()
        tot = len(series)
        probs = counts/tot
        return probs
    
    def probability(self, words):
        probability = 1.0
        for word in words:
            if word in self.mdl:
                probability *= self.mdl[word]
            else:
                probability = 0.0
        return float(probability)
        
    def sample(self, M):
        tokens = self.mdl.sample(n=M, weights=self.mdl.values, replace=True, random_state=1)
        randomized = ' '.join(tokens.index)
        return randomized


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


import numpy as np
import pandas as pd

class NGramLM(object):

    def __init__(self, N, tokens):
        self.N = N
        ngrams = self.create_ngrams(tokens)
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)
        
        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N - 1, tokens)

    def create_ngrams(self, tokens):
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            ngram = tuple(tokens[i : i + self.N])
            ngrams.append(ngram)
        return ngrams

    def train(self, ngrams):
        ngram_counts = {}
        n1_counts = {}
        for gram in ngrams:
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1
            prefix = gram[:-1]
            n1_counts[prefix] = n1_counts.get(prefix, 0) + 1
        
        final_list = []
        for gram, count in ngram_counts.items():
            prefix = gram[:-1]
            cond_prob = count / n1_counts[prefix] if n1_counts[prefix] != 0 else 0.0
            final_list.append({
                'ngram': gram,
                'n1gram': prefix,
                'prob': cond_prob
            })
        
        df = pd.DataFrame(final_list)[['ngram', 'n1gram', 'prob']]
        return df

    def probability(self, words):
     
        words = tuple(words)
        probs = []

        # Current model: probability for each n-gram in the input
        for i in range(len(words) - self.N + 1):
            current_ngram = tuple(words[i : i + self.N])
            matched = self.mdl[self.mdl['ngram'] == current_ngram]
            if not matched.empty:
                probs.append(matched['prob'].values[0])
            else:
                probs.append(0)

        curr_model = self.prev_mdl
        for k in range(self.N - 1, 1, -1):
            prefix = words[:k]
            matched = curr_model.mdl[curr_model.mdl['ngram'] == prefix]
            if not matched.empty:
                probs.append(matched['prob'].values[0])
            else:
                probs.append(0)
            curr_model = curr_model.prev_mdl

       
        if words[0] in curr_model.mdl.index:
            unigram_prob = curr_model.mdl.loc[words[0]]
            probs.append(unigram_prob)
        else:
            probs.append(0)

        return np.prod(probs)

    def sample(self, M):
       
        def nextToken(k, model):
            if k == 2:
                valid_ngrams = model.mdl[model.mdl['n1gram'] == ('\x02',)]
                if not valid_ngrams.empty:
                    return np.random.choice(valid_ngrams['ngram'], p=valid_ngrams['prob'])
                else:
                    return ('\x02', '\x03')
            token = nextToken(k - 1, model.prev_mdl)
            valid_ngrams = model.mdl[model.mdl['n1gram'] == token]
            if not valid_ngrams.empty:
                return np.random.choice(valid_ngrams['ngram'], p=valid_ngrams['prob'])
            else:
                return token + ('\x03',)
        
        tokens = list(nextToken(self.N, self))
        while len(tokens) < M:
            nextQuery = tuple(tokens[-(self.N - 1):])
            valid_ngrams = self.mdl[self.mdl['n1gram'] == nextQuery]
            if not valid_ngrams.empty:
                chosen = np.random.choice(valid_ngrams['ngram'], p=valid_ngrams['prob'])
                tokens.append(chosen[-1])
            else:
                tokens.append('\x03')
            if tokens[-1] == '\x03':
                break
        if tokens[-1] != '\x03':
            tokens.append('\x03')
        return ' '.join(tokens)