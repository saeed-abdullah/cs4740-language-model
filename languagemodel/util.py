# Miscellaneous utility functions.

def preprocess_text(inputfile, outputfile):
    """Performs preprocessing on raw text for creating vocabulary.

    This method performs sentence segmentation, tokenizer and
    WordNet lemmatization sequentially. For sentence segmentation,
    it uses Punkt sentence segmenter and for word tokenization
    nltk.word_tokenizer is used.

    It is assumed that there is a local copy for nltk data. For further
    details see, http://nltk.googlecode.com/svn/trunk/doc/howto/data.html
    
    After preprocessing, it writes back the resultant sentences
    to outputfile, where each line represents a single sentence.

    param:
    ----
    inputfile: Path to input file.
    outputfile: Path to output file.
    """

    import nltk

    # Load Punkt sentence segmenter model
    punkt_model = "nltk:tokenizers/punkt/english.pickle"
    sentence_tokenizer = nltk.data.load(punkt_model)

    # Initilize lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    with open(inputfile) as f:
        raw_data = f.read()

    # Sentence segmentation
    lines = sentence_tokenizer.tokenize(raw_data)

    with open(outputfile, "w") as f:
        for l in lines:
            # get words
            tokens = nltk.word_tokenize(l)

            # Doing lemmatization.
            normalized_tokens = [lemmatizer.lemmatize(t) for t in tokens]

            # Writing the sentence back where each word is separated
            # by space
            f.write(" ".join(normalized_tokens))
            f.write("\n")

