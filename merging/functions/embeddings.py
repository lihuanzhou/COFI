# importing libraries
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
import pandas as pd
import string
import numpy as np
import tensorflow_hub as hub
import fasttext
from scipy.spatial.distance import cosine

##### General #####
def get_cosine_similarity(embedding_1, embedding_2):
    # Compute cosine similarity between the original sentence embedding and the example sentence embedding
    similarity_score = cosine_similarity(embedding_1, embedding_2)
    
    # Print the similarity score
    return(similarity_score[0][0])

# TODO: this should go in the cleaning.py file (it's already there, so we should remove it from here
# and make sure that where this function is used is no longer imported from here but from cleaning.py)
def preprocess_text(text, remove_punctuation = True):
  # Implement your preprocessing logic here (e.g., lowercase conversion)

    new_string = text.lower()

    if remove_punctuation is True:
        # Create a translation table 
        translator = str.maketrans('', '', string.punctuation) 
        # Remove punctuation 
        new_string = new_string.translate(translator) 

    return new_string


##### BERT #####
def get_BERT_and_do_set_up():
    # do set up

    # Set a random seed
    random_seed = 42
    random.seed(random_seed)
    
    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # load BERT

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    return model, tokenizer


def get_BERT_embedding(text, model, tokenizer): 
# Tokenize and encode the example sentence
    example_encoding = tokenizer.batch_encode_plus(
        [text],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    example_input_ids = example_encoding['input_ids']
    example_attention_mask = example_encoding['attention_mask']
    
    # Generate embeddings for the example sentence
    with torch.no_grad():
        example_outputs = model(example_input_ids, attention_mask=example_attention_mask)
        example_sentence_embedding = example_outputs.last_hidden_state.mean(dim=1)

    return example_sentence_embedding

def embed_BERT_and_get_similarity(string_1, string_2, model, tokenizer):
    # check that the embeddings are valid
    

    return get_cosine_similarity(get_BERT_embedding(string_1, model, tokenizer), get_BERT_embedding(string_2, model, tokenizer))


#### Universal Encoder #####
def get_Universal_Encoder():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)

    return model

def embed_Universal_Encoder(input, model):
  return model(input)

def embed_Universal_Encoder_and_get_similarity(string_1, string_2, model):
    return np.inner(embed_Universal_Encoder([string_1], model), embed_Universal_Encoder([string_2], model))[0][0]


#### FastText ####
def get_FastText(bin_file):
    model = fasttext.load_model(bin_file)

    return model


def get_word_embedding_FastText(word, model):
  # Get the word vector from the model (might return None for unseen words)
    word_vector = model.get_word_vector(word)
    return word_vector

def embed_FastText_and_get_similarity(string1, string2, model):
    # Preprocess the strings (optional)
    processed_string1 = preprocess_text(string1)
    processed_string2 = preprocess_text(string2)

    # Get word embeddings (handle potential missing values)
    embedding1 = np.zeros(10)  # Initialize with zeros for unseen words
    for word in processed_string1.split():
        word_vector = get_word_embedding_FastText(word, model)
        if word_vector is not None:
            embedding1 += word_vector
    embedding1 /= len(processed_string1.split())  # Average word vectors

    # check that the embedding is valid
    all_zeros = not np.any(embedding1)
    if all_zeros is True:
        return -1000

    embedding2 = np.zeros(10)  # Initialize with zeros for unseen words
    for word in processed_string2.split():
        word_vector = get_word_embedding_FastText(word, model)
        if word_vector is not None:
            embedding2 += word_vector
    embedding2 /= len(processed_string2.split())  # Average word vectors

    # check that the embedding is valid
    all_zeros = not np.any(embedding2)
    if all_zeros is True:
        return -1000

    # Calculate cosine similarity
    
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity



