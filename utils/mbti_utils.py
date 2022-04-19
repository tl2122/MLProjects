import re
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence  import pad_sequences
from   nltk.stem.porter import PorterStemmer
from   nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def tokenize(posts, vocab_size):
    print("Starting tokenizer")
    allposts  = []

    for p1 in posts:
        for p2 in p1:
            allposts.append(p2)
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(allposts)
    posts_tk = tokenizer.texts_to_sequences(posts)

    post_size = [len(i) for i in posts]
    max_seq   = np.max(post_size)
    med_seq   = np.median(post_size)
    posts_tk  = pad_sequences(posts_tk, maxlen=max_seq, padding='post')
    print(10*"--",'\n',"Max Sequence Length",max_seq)
    print("Median Sequence Length",med_seq)
    print("Vocab Size",vocab_size)
    print(len(posts_tk),"tokenized")
    return posts_tk, max_seq


def makeModel(vocab_length,max_seq,embedding_dim, gru_units, makeBinary=True):
    inputs = tf.keras.Input(shape=(max_seq,))

    embedding = tf.keras.layers.Embedding(
                input_dim   = vocab_length,
                output_dim  = embedding_dim,
                input_length= max_seq
    )(inputs)

    gru = tf.keras.layers.Bidirectional(
          tf.keras.layers.GRU(
            units = gru_units,
            return_sequences=True
          )
    )(embedding)

    flatten = tf.keras.layers.Flatten()(gru)
    if makeBinary:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
    else:
        outputs = tf.keras.layers.Dense(16, activation='softmax')(flatten)

    model   = tf.keras.Model(inputs,outputs)
    return model



def getPosts(dataset):
    posts      = []
    ps         = PorterStemmer()
    #
    texts  = dataset['posts'].copy()
    labels = dataset['type'] .copy()
    unq_labels     = labels.unique()
    labels_dict    = {ilab:i for i,ilab in enumerate(unq_labels)}
    #
    texts      = [t.lower() for t in texts]
    texts      = [t.split("|||") for t in texts]
    #
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    stop_words.remove('no')

    newlabels = []
    for idx,text in enumerate(texts):
        lb_ = labels[idx]
        for t in text:
            t = re.sub("http\S+","",t) # remove https
            t = re.sub("www\S+" ,"",t) # remove www
            t = re.sub(" +"," ",t)     # remove blank space
            t = t.split()              #remove stopwords
            t = [ps.stem(word) for word in t if not word in set(stop_words)]
            t = ' '.join(t)
            if len(t)>0:
                posts.append(t)
                newlabels.append(lb_)

    print (len(posts),"posts added.")
    return posts,newlabels,labels_dict


def convert_to_one_hot(Y, C):
    #Y = np.eye(C)[Y.reshape(-1)].T
    print (Y)
    print (C)
    Y = np.eye(C)[Y]
    return Y
                
