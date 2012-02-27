# Probability Distributions for language models.

class ProbabilityDistribution(object):
    """Base unsmoothed probability distribution class.

    This class builds unsmoothed probability distribution from
    the frequency count. For representing the conditional probability
    P(U|V), it uses the ngram --- 'V U' as the key. So, the probability of
    'am' given 'I, has been stored by using the key 'I am' in the
    dictionary.
    """

    def __init__(self, vocabulary):
        """Initiates the class.
        param
        ----
        vocabulary: The size of vocabulary.
        """
        self._vocabulary_count = vocabulary
        self._probability_distribution = {}

    def build_probability(self, ngram, subgram):
        """Builds probability distribution for given ngrams.

        The probability is determined by the eqn. 4.15 ---
        frequency of that n-gram/frequncy of (n-1)-gram with common prefix.

        The n-grams are being used as key for storing.

        params
        ----
        ngram: The dictionary containing the frequency of ngrams --- see 
        Ngrams.py for more details.
        subgram: The dictionary containing the frequency of (n-1) grams
        --- see Ngrams.py for more details.
        """

        for k, f in ngram.iteritems():
            # Get (n-1) grams with common prefix.
            sub_gram = " ".join(k.split()[:-1])
            sub_gram_f = subgram[sub_gram]

            self._probability_distribution[k] = float(f)/sub_gram_f

    def get_probability(self, ngram):
        """Returns of the probability.

        param
        ----
        ngram: To determine the conditional probability of P(V|U), the ngram
        should be string containing 'U V'.

        return
        ----
        Returns the probability.
        """
        if self._probability_distribution.has_key(ngram):
            return self._probability_distribution[ngram]
        else:
            return 0

class LaplaceSmoothedDistribution(ProbabilityDistribution):
    """Probability distribution with Laplace Smoothing.

    This class retains a reference to the frequency count of N-1 grams
    after building probability distribution so that it can handle
    P(W_n| W_1 W_2 ... W_(n-1)) where N-gram W_n W_(n-1) .. W_1
    was not in the train set.
    """

    def __init__(self, vocabulary):
        """Initiates the class.
        param
        ----
        vocabulary: The size of vocabulary.
        """

        super(LaplaceSmoothedDistribution, self).__init__(vocabulary)

    def build_probability(self, ngram, subgram):
        self._subgram = subgram
        for k, f in ngram.iteritems():
            # Get (n-1) grams with common prefix.
            sub_gram = " ".join(k.split()[:-1])
            sub_gram_f = subgram[sub_gram]

            prob = float(f + 1)/(sub_gram_f + self._vocabulary_count)

            self._probability_distribution[k] = prob

    def get_probability(self, ngram):
        """Returns of the probability.

        param
        ----
        ngram: To determine the conditional probability of P(V|U), the ngram
        should be string containing 'U V'.

        return
        ----
        Returns the probability.
        """
        if self._probability_distribution.has_key(ngram):
            return self._probability_distribution[ngram]
        else:
            # Get (n-1) grams with common prefix.
            sub_gram = " ".join(ngram.split()[:-1])

            if self._subgram.has_key(sub_gram):
                sub_gram_f = self._subgram[sub_gram]
            else:
                # This should not happen. And, allowing probability
                # for sub-grams which did not appear in training set
                # might resulting in total Probability being greater
                # than 1.0. For example, for bigrams, the vocabulary
                # size should equal to the number of unigrams, but
                # we are allowing 1/V probability for tokens which
                # are not in vocabulary.
                sub_gram_f = 0

            prob = 1.0 / (sub_gram_f + self._vocabulary_count)

            return prob




