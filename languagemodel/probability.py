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

import math
class GoodTuringDistribution(object):

    def __init__(self):
        self._pzero = 0
        self._prob_values = {}

    def build_probability(self, ngrams):
        frequency_of_frequency, total_count = \
                GoodTuringDistribution._get_frequency_count(ngrams)

        self._pzero = float(frequency_of_frequency[1])/total_count

        z_vals = GoodTuringDistribution._get_z_values(frequency_of_frequency)
        alpha, beta = GoodTuringDistribution._perform_linear_regression(
                z_vals)
        r_star = GoodTuringDistribution._get_r_star_values(
                frequency_of_frequency, alpha, beta)

        self._set_probability(r_star, frequency_of_frequency, ngrams)


    @staticmethod
    def _get_frequency_count(ngrams):
        frequency_of_frequency = {}
        total_count = 0
        for count in ngrams.itervalues():
            if not frequency_of_frequency.has_key(count):
                frequency_of_frequency[count] = 0

            frequency_of_frequency[count] += 1
            total_count += count

        return frequency_of_frequency, total_count

    @staticmethod
    def _get_z_values(frequency_of_frequency):
        z_vals = {}
        sorted_r = sorted(frequency_of_frequency.keys())
        for index, n_r in enumerate(sorted_r):
            if index == 0:
                q = 0
            else:
                q = sorted_r[index - 1]
            if index == len(sorted_r) - 1:
                t = 2 * n_r - q
            else:
                t = sorted_r[index + 1]

            z_vals[n_r] = 2. * frequency_of_frequency[n_r]/(t - q)

        return z_vals

    @staticmethod
    def _perform_linear_regression(z_vals):
        import numpy as np
        x_vals = np.log(z_vals.keys())
        y_vals = np.log(z_vals.values())

        # Convert to coefficient matrix
        coeff_matrix = np.vstack([x_vals, np.ones(len(x_vals))]).T
        m, c = np.linalg.lstsq(coeff_matrix, y_vals)[0]

        return m, c

    @staticmethod
    def _get_smoothed_value(val, alpha, beta):
        return math.exp(alpha * math.log(val) + beta)

    @staticmethod
    def _get_r_star_values(frequency_of_frequency, alpha, beta,
            confid_factor=1.96):

        r_star = {}
        sorted_r = sorted(frequency_of_frequency.keys())

        is_using_turing_estimate = True

        for r in sorted_r:
            smoothed_r = GoodTuringDistribution._get_smoothed_value(r,
                    alpha, beta)
            smoothed_r_1 = GoodTuringDistribution._get_smoothed_value(r + 1,
                    alpha, beta)
            y = (r + 1) * smoothed_r_1 / smoothed_r
            if r + 1 not in sorted_r:
                is_using_turing_estimate = False

            if is_using_turing_estimate:
                val_n_r = frequency_of_frequency[r]
                val_n_r_1 = frequency_of_frequency[r + 1]
                x = float(r + 1) * val_n_r_1 / val_n_r

                confid_1 = float(val_n_r) / (val_n_r_1**2) * \
                        (1. + float(val_n_r_1) / val_n_r)
                t = confid_factor * math.sqrt(((r + 1)**2) * confid_1)

                if math.fabs(x - y) <= t:
                    is_using_turing_estimate = False
                    r_star[r] = y
                else:
                    r_star[r] = x
            else:
                r_star[r] = y

        return r_star

    def _set_probability(self, r_star, frequency_of_frequency, ngrams):
        total_val = 0
        for r, r_s in r_star.iteritems():
            total_val += frequency_of_frequency[r] * r_s

        for ngram, count in ngrams.iteritems():
            self._prob_values[ngram] = (1 - self._pzero) * \
                    float(r_star[count]) / total_val

    def get_probability(self, ngram):
        if self._prob_values.has_key(ngram):
            return self._prob_values[ngram]
        else:
            return self._pzero

