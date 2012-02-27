# Language modeling using N-grams

import os

class NGram:
    """
    Builds a N-Grams based language model.

    For building a model, this class calculates frequency of
    all the N-grams and N-1 grams (subgrams) in the corpus. 
    """

    # Marker for start and end of the sentence.
    _START_MARKER_ = "_START_"
    _END_MARKER_ = "_END_"

    def __init__(self, inputfile, N=3):
        """Initializes the instance.

        param:
        ----
        inputfile: Path of input file. This file should contain each
            sentence per line. For more details, see util.preprocess_text
        N: Size of Markov memory, must be greater than 1.
        """
        if N <= 1:
            raise ValueError("Can not create unigram")

        self._grams_count = {}
        self._subgrams_count = {}

        self._inputfile = inputfile
        self._window_size = N

    def get_ngrams(self):
        """Returns all N-grams"""
        return self._grams_count.keys()

    def get_subgrams(self):
        """Returns all N-1 grams."""
        return self._subgrams_count.keys()

    def get_ngrams_frequency(self, ngram):
        """Returns frequency count for n-gram"""
        if ngram not in self._grams_count.keys():
            return 0
        else:
            return self._grams_count[ngram]

    def get_subgrams_frequency(self, subgram):
        """Returns frequency count for n-1 gram"""
        if subgram not in self._subgrams_count.keys():
            return 0
        else:
            return self._subgrams_count[subgram]

    def build_ngrams(self):
        """
        Calculates the frequncy of N-grams and N-1 subgrams.
        """

        with open(self._inputfile) as f:
            for l in f:
                line = l.strip().split()

                # Adding the end marker at the end of current line.
                line.extend([NGram._END_MARKER_])

                # Fill required starters.
                # For example given the sentence, "I am walking .",
                # the trigrams are: '<s> <s> I', '<s> I am', 'I am walking',
                # 'am walking .', and, 'walking . </s>'.
                for index in range(self._window_size - 1, 0, -1):
                    starters = [NGram._START_MARKER_] * index
                    starters.extend(line[:self._window_size - index])

                    key = " ".join(starters)
                    if key not in self._grams_count.keys():
                        self._grams_count[key] = 0
                    self._grams_count[key] += 1

                    # Fill start for N-1 grams
                    if index > 1:
                        # For N-1 grams, last iteration should not be used.
                        key = " ".join(starters[1:])
                        if key not in self._subgrams_count.keys():
                            self._subgrams_count[key] = 0
                        self._subgrams_count[key] += 1


                # All the starters have been produced.
                # Given the L-words, the number of N-1 grams we
                # can produce is L - ((N-1)-1).
                for index in range(0, len(line) - (self._window_size - 2)):
                    # Get N-1 gram
                    key = " ".join(
                            line[index : index + self._window_size - 1])

                    if key not in self._subgrams_count:
                        self._subgrams_count[key] = 0

                    self._subgrams_count[key] += 1

                    # Get N-gram
                    if index + self._window_size <= len(line):
                        key = " ".join(
                                line[index : index + self._window_size])
                        if key not in self._grams_count:
                            self._grams_count[key] = 0
                        self._grams_count[key] += 1

    def dump_data(self, output_dir):
        """Dumps ngram and subgram counts in json format.

        param
        ----
        output_dir: The path of output dir.
        """
        import json

        output_path_pattern = output_dir + "{0}_N_{1}.json"

        with open(output_path_pattern.format("ngrams",
            self._window_size), "w") as f:
            json.dump(self._grams_count, f)

        with open(output_path_pattern.format("subgrams",
            self._window_size), "w") as f:
            json.dump(self._subgrams_count, f)

def calculates_frequency(files):
    """Util function for calculating data frequency."""
    import glob

    for f in files:
        d, s = os.path.split(f)
        outdir = os.path.join(d, "json_train/")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        language_model = NGram(f, N=2)
        language_model.build_ngrams()
        language_model.dump_data(outdir)

