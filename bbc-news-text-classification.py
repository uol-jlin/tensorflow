import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("./data/bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"Each data point looks like this:\n\n{csvfile.readline()}")     

def remove_stopwords(sentence):
    """
    Removes a list of stopwords
    
    Args:
        sentence (string): sentence to remove the stopwords from
    
    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    
    sentence = sentence.lower()
    split_sentence = sentence.split(" ")
    cleaned_sentence = [word for word in split_sentence if word not in stopwords]
    
    return " ".join(cleaned_sentence)
    
def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file
    
    Args:
        filename (string): path to the CSV file
    
    Returns:
        sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for row in reader:
            labels.append(remove_stopwords(row[0]))
            sentences.append(remove_stopwords(row[1]))
    return sentences, labels
    
def fit_tokenizer(sentences):
    """
    Instantiates the Tokenizer class
    
    Args:
        sentences (list): lower-cased sentences without stopwords
    
    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    # Instantiate the Tokenizer class by passing in the oov_token argument
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def get_padded_sequences(tokenizer, sentences):
    """
    Generates an array of token sequences and pads them to the same length
    
    Args:
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        sentences (list of string): list of sentences to tokenize and pad
    
    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the post padding strategy
    padded_sequences = pad_sequences(sequences, padding='post')

    return padded_sequences

def tokenize_labels(labels):
    """
    Tokenizes the labels
    
    Args:
        labels (list of string): labels to tokenize
    
    Returns:
        label_sequences, label_word_index (list of string, dictionary): tokenized labels and the word-index
    """    
    # Instantiate the Tokenizer class
    label_tokenizer = Tokenizer()
    
    # Fit the tokenizer to the labels
    label_tokenizer.fit_on_texts(labels)

    # Save the word index
    label_word_index = label_tokenizer.word_index
    
    # Save the sequences
    label_sequences = label_tokenizer.texts_to_sequences(labels)
    
    return label_sequences, label_word_index
    
