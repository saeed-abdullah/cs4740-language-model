# Test cases for util.py

from .. import util

def test_preprocess_text():
    inputfile = "/tmp/sentence_converter.txt"
    outputfile = "/tmp/sentence_converted.txt"

    # For this text, 're shows up as a single token.
    text = """You are not your job. You're not how much money you have in 
    the bank. You're not the car you drive. You're not the contents of 
    your wallet. You're not your fucking khakis. You're the all-singing,
    all-dancing crap of the world."""

    expected = ["You are not your job .",
        "You 're not how much money you have in the bank .",
        "You 're not the car you drive .",
        "You 're not the content of your wallet .",
        "You 're not your fucking khaki .",
        "You 're the all-singing , all-dancing crap of the world ."]

    with open(inputfile, "w") as f:
        f.write(text)

    util.preprocess_text(inputfile, outputfile)

    with open(outputfile) as f:
        for i, l in enumerate(f):
            line = l.strip()
            assert line == expected[i]


def test_ngrams_from_line():

    line = "I am walking ."
    expected_ngrams = ["_START_ _START_ I", "_START_ I am",
            "I am walking", "am walking .", "walking . _END_"]
    expected_subgrams = ["_START_ I", "I am", "am walking",
            "walking .", ". _END_"]

    actual_ngrams = util.get_ngrams_from_line(line, 3, "_START_",
            "_END_")

    actual_subgrams = util.get_ngrams_from_line(line, 2, "_START_",
            "_END_")

    assert expected_ngrams == actual_ngrams
    assert expected_subgrams == actual_subgrams


