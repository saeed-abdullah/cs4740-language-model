# Test cases for NGram

from .. import NGrams

import unittest

class TestNGrams(unittest.TestCase):

     def test_build_ngrams(self):
        inputfile = "/tmp/1"
        line = "I am walking ."
        with open(inputfile, "w") as f:
            f.write(line)
        language_model = NGrams.NGram(inputfile)

        language_model.build_ngrams()
        expected_ngrams = ["_START_ _START_ I", "_START_ I am",
                "I am walking", "am walking .", "walking . _END_"]
        expected_subgrams = ["_START_ I", "I am", "am walking",
                "walking .", ". _END_"]

        for ngram in expected_ngrams:
            assert language_model.get_ngrams_frequency(ngram) == 1

        for subgram in expected_subgrams:
            assert language_model.get_subgrams_frequency(subgram) == 1

        # Testing for lines which
        # contains less than N-1 tokens
        with open(inputfile, "a+") as f:
            f.write("\n")
            f.write("I")
            f.write("\n")

        # Input file now contains three lines
        # I am walking
        # I
        #
        language_model = NGrams.NGram(inputfile)
        language_model.build_ngrams()

        assert language_model.get_ngrams_frequency("_START_ I _END_") == 1
        assert language_model.get_subgrams_frequency("_START_ I") == 2

if __name__ == "__main__":
    unittest.main()

