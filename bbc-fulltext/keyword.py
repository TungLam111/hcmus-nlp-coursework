import sys
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math

def preprocess(document):

    # lowercase
    lower_case_doc = document.lower()

    # remove punctuation and numbers
    remove_punc_doc =  re.sub(r'[^A-Za-z\s]', '', lower_case_doc)

    # remove whitespace and tokenize
    tokenized_doc = re.split('\s+', remove_punc_doc.strip())

    # remove stopwords
    stopwords_corpus = set(stopwords.words('english'))
    remove_stopw_doc = [x for x in tokenized_doc if x not in stopwords_corpus]
    
    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_doc = [wordnet_lemmatizer.lemmatize(x) for x in remove_stopw_doc]

    return lemma_doc

def calculate_idf(corpus):
    freq_dict = {}
    document_tf = []

    N = len(corpus)

    for document in corpus:
        document_tf.append({})
        for word in document:
            if document_tf[len(document_tf)-1].get(word) is None:
                document_tf[len(document_tf)-1][word] = calculate_tf(word, document)
        
            if freq_dict.get(word) is None:
                freq_dict[word] = 1
            else:
                freq_dict[word] = freq_dict[word] + 1

    for key in freq_dict:
        freq_dict[key] = math.log(N / (1 + freq_dict[key]))

    '''
    document_tf: [{'word1': }]
    '''

    return freq_dict, document_tf

def calculate_tf(word, words_in_document):
    count = 0
    for item in words_in_document:
        if item == word:
            count = count + 1
    return count / len(words_in_document)


def process_subfolder(data_path):
    documents = []
    list_dir = os.listdir(data_path)
    for fname in list_dir:
        if fname[0] == '.':
            continue
        path = os.path.join(data_path,fname)
        f = open(path,'r')
        processed_doc = preprocess(f.read())

        documents.append(processed_doc)

    assert len(documents) == len(list_dir)

    return documents, list_dir


def process(corpus, top_key):

    freq_dict, document_tf = calculate_idf(corpus)
    res = []
    for doc in document_tf:
        for key in doc:
            doc[key] = doc[key] * freq_dict[key]

        res.append(sorted(doc.items(), key=lambda kv: kv[1], reverse=True)[:top_key])
    return res

def main(data_path):
    corpus, list_dir = process_subfolder(data_path)
    res = process(corpus, 5)

    with open('output.txt', 'w') as f:
        for idx, doc in enumerate(res):
            f.write(list_dir[idx])
            f.write(' ')
            
            for value in doc:
                f.write('{} '.format(value[0])) 
            if idx != len(list_dir)-1:
                f.write('\n')
        f.close()

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tkeyword.py <data_dir>')
        sys.exit(0)
    main(sys.argv[1])
