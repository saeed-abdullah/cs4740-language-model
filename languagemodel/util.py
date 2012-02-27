# Miscellaneous utility functions.

from NGrams import NGram

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
            # by a single space
            f.write(" ".join(normalized_tokens))
            f.write("\n")


def get_ngrams_from_line(sentence, window_size, start_symbol, end_symbol):
    """Retrieves NGrams from given line.

    It uses a sliding window of length N for producing all possible
    N-grams.

    params:
    sentence: Line where each word is separated by whitespace.
    window_size: The value of N in N-grams.
    start_symbol: Start symbol to use, see NGram._START_.
    end_symbol: End symbol to use, see NGram._END_.

    return
    ----
    A list of ngrams in the order of the appearance.
    """

    ngrams = []
    line = sentence.strip().split()

    # Adding the end marker at the end of current line.
    line.extend([end_symbol])

    # Fill required starters.
    # For example given the sentence, "I am walking .",
    # the trigrams are: '<s> <s> I', '<s> I am', 'I am walking',
    # 'am walking .', and, 'walking . </s>'.
    for index in range(window_size - 1, 0, -1):
        starters = [start_symbol] * index
        starters.extend(line[:window_size - index])
        ngrams.append(" ".join(starters))

    # All the starters have been produced.
    # Given the L-words, the number of N-1 grams we
    # can produce is L - (N-1).
    for index in range(0, len(line) - (window_size - 1)):
        ngrams.append(" ".join(line[index : index + window_size]))

    return ngrams


def calculate_perplexity(filename, prob_distribution, window_size):
    """Calculates the perplexity of test data.

    It returns the log of the eqn. (4.18) for better accuracy.

    param
    ----
    filename: Input file name. It is assumed that the file has been
        preprocessed so that each line contains a single complete sentence.
        For more details, see preprocess_text.
    prob_distribution: Distribution from training data, see
        LaplaceSmoothedDistribution for more details.
    window_size: The N of N-grams.

    returns
    ----
    Log value of perplexity.
    """
    prob_sum = 0
    token_count = 0

    import math

    with open(filename) as f:
        for line in f:
            ngrams = get_ngrams_from_line(line, window_size,
                    NGram._START_MARKER_, NGram._END_MARKER_)

            for n in ngrams:
                prob = prob_distribution.get_probability(n)
                prob_sum += math.log(prob)

            # As specified in book (pg. 96), that token count
            # should not include start symbol.
            token_count += len(ngrams) - (window_size - 1)

    return -1./token_count * prob_sum
