# Test cases for probability distribution.
from .. import probability

import unittest

class TestProbability(unittest.TestCase):

    @classmethod
    def setUpClass(klass):
        # For following sentences:
        # I am . I do .
        ngrams = {"_START_ I": 2, "I am": 1, "am .": 1, ". _END_": 2,
                "I do": 1, "do .": 1}
        subgrams = {"I": 2, "am": 1, "do": 1, ".": 2, "_END_": 2,
                "_START_": 2}

        klass._ngrams = ngrams
        klass._subgrams = subgrams

    def test_probability_distribution(self):
        prob = probability.ProbabilityDistribution(
                len(TestProbability._subgrams))

        prob.build_probability(TestProbability._ngrams,
                TestProbability._subgrams)

        self.assertAlmostEqual(prob.get_probability("I am"), 0.5)
        self.assertAlmostEqual(prob.get_probability("am do"), 0)

    def test_laplace_smoothed_distribution(self):
        prob = probability.LaplaceSmoothedDistribution(
                len(TestProbability._subgrams))

        prob.build_probability(TestProbability._ngrams,
                TestProbability._subgrams)

        self.assertAlmostEqual(prob.get_probability("I am"), 2./8)
        self.assertAlmostEqual(prob.get_probability("blah do"), 1./6)

    def test_good_turing_distribution(self):
        prob = probability.GoodTuringDistribution()
        prob.build_probability(TestProbability._ngrams)

        self.assertAlmostEqual(prob.get_probability("I am"), 0.0625)
        self.assertAlmostEqual(prob.get_probability("blah do"), 1./2)
