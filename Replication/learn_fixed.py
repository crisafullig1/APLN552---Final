import copy
import math
import numpy
import scipy
import re
import sys
import os

import constants as CONST
from category_cleaned import *
import input
import wmmapping
import statistics
import evaluate
import wgraph_cleaned


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

"""
learn.py
"""

def postag_key_match(word, key):
    """
    Return whether the part-of-speech tag of `word` is consistent with `key`.
    Treats missing tags as CONST.OTH-compatible.
    """
    if key == CONST.ALL:
        return True

    tag = postag(word)
    return key == tag or (key == CONST.OTH and tag not in {CONST.V, CONST.N, ''})


def postag(word):
    """
    Return the part-of-speech tag of `word`, where word is of the form 'token:tag'.
    If no tag is present, return an empty string.
    """
    parts = re.findall(r"([^:]+):", word + ":")
    return parts[1] if len(parts) > 1 else ''


def postag_count(words, postags):
    """
    Return the number of words in `words` whose part-of-speech tag is in `postags`.
    """
    return sum(1 for word in words if postag(word) in postags)


class Learner:
    """
    Encapsulate all learning and model updating functionality.
    """
    
    def __init__(self, lexicon_path, config, stopwords=[]):
        if not os.path.exists(lexicon_path):
            print("Initialization Error -- Lexicon does not exist : " + str(lexicon_path))
            sys.exit(2)
            
        if config is None:
            print("Initialization Error -- Config required")
            sys.exit(2)
        
        self._beta = config.param_float("beta")
        if self._beta < 0:
            print("Config Error [beta] Must be non-zero positive : " + str(self._beta))
            sys.exit(2)
        
        self._lambda = config.param_float("lambda")
        self._power = config.param_int("power")
        if self._lambda > 1 and self._power <= 0:
            print("Config Error [lambda] [power]")
            print("\t lambda: " + str(self._lambda) + ", power: " + str(self._power))
            sys.exit(2)
        
        self._alpha = config.param_float("alpha")
        if self._alpha == 0:
            print("Config Warning [alpha] No alpha smoothing")
        if self._alpha < 0:
            print("Config Error [alpha] Must be positive : " + str(self._alpha))
            sys.exit(2)
        
        self._epsilon = config.param_float("epsilon")
        if self._epsilon <= 0:
            print("Config Error [epsilon] Must be non-zero positive : " + str(self._epsilon))
        
        self._theta = config.param_float("theta")
        if self._theta < 0:
            print("Config Error [theta] Must be non-zero positive : " + str(self._theta))
            sys.exit(2)
        
        self._simtype = config.param("simtype")
        if self._simtype not in CONST.ALL_SIM_TYPES:
            print("Config Error [simtype] Invalid simtype : " + str(self._simtype))
            sys.exit(2)

        self.alignment_method = config.param_int("alignment-method")
        if self.alignment_method < 0:
            self.alignment_method = 0

        self._dummy = config.param_bool("dummy")
        self._forget_flag = config.param_bool("forget")
        self._novelty_flag = config.param_bool("novelty")
        self._forget_decay = config.param_float("forget-decay")
        self._novelty_decay = config.param_float("novelty-decay")

        if self._forget_flag and self._forget_decay <= 0:
            print("Config Error [forget-decay] Must be non-zero positive : " + str(self._forget_decay))
            sys.exit(2)
        if self._novelty_flag and self._novelty_decay <= 0:
            print("Config Error [novelty-decay] Must be non-zero positive : " + str(self._novelty_decay))
            sys.exit(2)

        self._assoc_type = config.param("assoc-type")
        if self._assoc_type not in CONST.ALL_ASSOC_TYPES:
            print("Config Error [assoc-type] Invalid assoc-type : " + str(self._assoc_type))
            sys.exit(2)

        self._category_flag = config.param_bool("category")
        self._stats_flag = config.param_bool("stats")
        self._context_stats_flag = config.param_bool("context-stats")
        
        self._postags = set()
        for tag in ["tag1", "tag2", "tag3"]:
            if config.has_param(tag):
                if config.param(tag) not in CONST.ALL_TAGS:
                    print(f"Config Error [{tag}] Invalid : " + config.param(tag))
                    sys.exit(2)
                self._postags.add(config.param(tag))

        if self._stats_flag:
            self._wordsp = statistics.WordPropsTable(config.param("word-props-name"))
            self._timesp = statistics.TimePropsTable(config.param("time-props-name"))
        
        if self._context_stats_flag:
            smoothing = config.param_float("familiarity-smoothing")
            if smoothing < 0:
                print("Config Error [familiarity-smoothing] Must be positive : " + str(smoothing))
                sys.exit(2)
            fam_measure = config.param("familiarity-measure")
            if fam_measure not in CONST.ALL_FAM_MEASURES:
                print("Config Error [familiarity-measure] Invalid : " + str(fam_measure))
                sys.exit(2)
            f_name = config.param("context-props-name")
            self._contextsp = statistics.ContextPropsTable(f_name, smoothing, fam_measure)
            self._contextsp._aoe_normalization = config.param_int("age-of-exposure-norm")

        self._tasktype = config.param("tasktype")
        self._minfreq = config.param_int("minfreq")
        self._record_itrs = config.param_int("record-iterations")
        self._maxtime = config.param_int("maxtime")
        self._maxlearned = config.param_int("maxlearned")
        self._remove_singletons = config.param_bool("remove-singleton-utterances")

        self._gold_lexicon = input.read_gold_lexicon(lexicon_path, self._beta)
        self._all_features = input.all_features(lexicon_path)
        print("number of Gold Features", len(self._all_features))

        self._learned_lexicon = wmmapping.Lexicon(self._beta, self._gold_lexicon.words())
        self._aligns = wmmapping.Alignments(self._alpha)

        self._time = 0
        self._vocab = set()
        self._features = set()
        self._acquisition_scores = {}
        self._last_time = {}
        self._acq_score_list = {}

        self._stopwords = stopwords
        self._wnlabels = WordnetLabels() 
        self._context = []

        self._grow_graph_flag = config.param_bool("semantic-network")

        def init_words_graph(self, hubs_num, sim_threshold, hub_type, coupling, lambda0, a0, miu0, sigma0, sampling_method):
            self._words_graph = wgraph.WordsGraph(hubs_num, sim_threshold, hub_type,
                                              coupling, lambda0, a0, miu0, sigma0,
                                              sampling_method)

    def reset(self):
        if self._stats_flag:
            self._wordsp.reset()
            self._timesp.reset()

        if self._context_stats_flag:
            self._contextsp.reset()

        self._learned_lexicon = wmmapping.Lexicon(self._beta, self._gold_lexicon.words())
        self._aligns = wmmapping.Alignments(self._alpha)

        self._time = 0
        self._vocab = set()
        self._features = set()
        self._acquisition_scores = {}
        self._last_time = {}

    def get_lambda(self):
        if self._lambda < 1 and self._lambda > 0:
            return self._lambda
        return 1.0 / (1 + self._time ** self._power)

    def learned_lexicon(self):
        return copy.deepcopy(self._learned_lexicon)

    def avg_acquisition(self, words, key):
        total = 0.0
        vsize = 0
        for word in words:
            if postag_key_match(word, key):
                total += self.acquisition_score(word)
                vsize += 1
        return 0.0 if vsize == 0 else total / float(vsize)

    def acquisition_score(self, word):
        if self._forget_flag or word not in self._acquisition_scores:
            self.calculate_acquisition_score(word)
        return self._acquisition_scores[word]

    def calculate_acquisition_score(self, word):
        if self._forget_flag:
            self.update_meaning_prob(word)
        true_m = self._gold_lexicon.meaning(word)
        lrnd_m = self._learned_lexicon.meaning(word)
        sim = evaluate.calculate_similarity(self._beta, lrnd_m, true_m, self._simtype)
        self._acquisition_scores[word] = sim
        return sim

    def calculate_prob_meaning(self, word, meaning, std):
        prob_meaning = 1.0
        for feature in meaning.seen_features():
            mu = self._learned_lexicon.meaning(word).prob(feature)
            prob_meaning *= scipy.stats.norm(mu, std).pdf(meaning.prob(feature))
        return prob_meaning

    def calculate_referent_prob(self, word, meaning, std):
        numerator = self.calculate_prob_meaning(word, meaning, std)
        numerator *= self._wordsp.frequency(word)

        denom = 0.0
        for other_word in self._wordsp.all_words(0):
            denom += self.calculate_prob_meaning(other_word, meaning, std) * self._wordsp.frequency(other_word)

        return numerator / denom if denom > 0 else 0.0

    def update_meaning_prob(self, word, time=-1):
        """
        Update the meaning probabilities of word in this learner's lexicon.
        This is done by calculating the association between this word and all
        encountered features - p(f|w) - then normalizing to produce a
        distribution.
        """
        if time == -1:
            time = self._time

        Lambda = self.get_lambda()
        associations = {}
        sum_assoc = 0.0

        for feature in self._features:
            assoc = self.association(word, feature, time)
            associations[feature] = assoc
            sum_assoc += assoc

        sum_assoc += self._beta * Lambda  # Smoothing

        for feature in self._features:
            meaning_prob = (associations[feature] + Lambda) / sum_assoc
            self._learned_lexicon.set_prob(word, feature, meaning_prob)

        prob_unseen = Lambda / sum_assoc
        self._learned_lexicon.set_unseen(word, prob_unseen)

    def association(self, word, feature, time):
        """
        Return the association score between word and feature.
        If SUM is the association type, the total alignment probabilities
        over time of word being aligned with feature is the association.

        If ACT is the association type, an activation function using this
        learner's forget_decay value is used to calculate the association.
        """
        if self._assoc_type in (CONST.SUM, CONST.DEC_SUM):
            return self._aligns.sum_alignments(word, feature)

        if self._assoc_type == CONST.ACT:
            aligns = self._aligns.alignments(word, feature)
            assoc_sum = 0.0

            for t_pr, val in aligns.items():
                align_decay = self._forget_decay / val
                assoc_sum += val / math.pow(time - t_pr + 1, align_decay)

            return math.log1p(assoc_sum)
    def novelty(self, word):
        """
        Return the novelty decay coefficient of a word based on the last time it 
        was encountered and the learner's novelty_decay base value.
        """
        last_time = self._last_time.get(word, 0)

        if last_time == 0:
        
            return 1.0

        delta_time = float(self._time - last_time) + 1
        denom = pow(delta_time, self._novelty_decay)
        return 1.0 - (1.0 / denom)

    def calculate_alignments(self, words, features, outdir, category_learner=None):
        """
        Update the alignments for each combination of word-feature pairs from
        the list `words` and set `features`.
        """
        if self._forget_flag:
            print("forget flag")
            for word in words:
                self.update_meaning_prob(word)

        category_flag = self._category_flag and category_learner is not None
        category_probs = {}
        if category_flag:
            category_probs = self.calculate_category_probs(words, features, category_learner)

        w_r_alignment = {}

        if self.alignment_method == 0:
            for feature in features:
                denom = sum(self._learned_lexicon.prob(word, feature) for word in words)
                category_denom = sum(category_probs[word][feature] for word in words) if category_flag else 0

                denom += self._alpha * self._epsilon
                category_denom += self._alpha * self._epsilon

                for word in words:
                    alignment = (self._learned_lexicon.prob(word, feature) + self._epsilon) / denom
                    weight = self._wordsp.frequency(word) / (1.0 + self._wordsp.frequency(word))

                    if category_flag:
                        category_prob = category_probs[word][feature]
                        factor = (category_prob + self._epsilon) / category_denom
                        alignment = weight * alignment + (1 - weight) * factor

                    if self._novelty_flag:
                        alignment *= self.novelty(word)

                    if self._assoc_type == CONST.DEC_SUM:
                        self._aligns.add_decay_sum(word, feature, self._time, alignment, self._forget_decay)
                    else:
                        self._aligns.add_alignment(word, feature, self._time, alignment)

        elif self.alignment_method in {1, 2, 3, 4, 5}:
            sim_score = {}
            w_denom = {}
            r_denom = {}
            wr_denom = 0.0

            for referent in features:
                r_index = features.index(referent)
                r_denom[r_index] = 0.0

                for word in words:
                    score = evaluate.sim_cosine_word_ref(self._beta, self._learned_lexicon.meaning(word), referent)
                    sim_score[(word, r_index)] = score
                    r_denom[r_index] += score
                    w_denom[word] = w_denom.get(word, 0.0) + score
                    wr_denom += score

            for word in words:
                for referent in features:
                    r_index = features.index(referent)
                    if self.alignment_method == 1:
                        denom = r_denom[r_index]
                        w_r_alignment[(word, r_index)] = sim_score[(word, r_index)] / denom if denom else 0.0
                    elif self.alignment_method == 2:
                        denom = sum(sim_score[(w, r_index)] for w in words)
                        w_r_alignment[(word, r_index)] = sim_score[(word, r_index)] / denom if denom else 0.0
                    elif self.alignment_method == 3:
                        w_r_alignment[(word, r_index)] = sim_score[(word, r_index)]
                    elif self.alignment_method == 4:
                        denom = w_denom[word] * r_denom[r_index]
                        w_r_alignment[(word, r_index)] = sim_score[(word, r_index)] / denom if denom else 0.0
                    elif self.alignment_method == 5:
                        denom = w_denom[word] * r_denom[r_index]
                        w_r_alignment[(word, r_index)] = (wr_denom * sim_score[(word, r_index)]) / denom if denom else 0.0

            for feature in set(flatten(features)):
                for word in words:
                    alignment = max(
                        (w_r_alignment[(word, i)] for i in range(len(features)) if feature in features[i]),
                        default=0.0
                    )

                    if category_flag:
                        weight = self._wordsp.frequency(word) / (1.0 + self._wordsp.frequency(word))
                        category_prob = category_probs[word][feature]
                        category_denom = sum(category_probs[word][feature] for word in words) + self._alpha * self._epsilon
                        factor = (category_prob + self._epsilon) / category_denom
                        alignment = weight * alignment + (1 - weight) * factor

                    if self._novelty_flag:
                        alignment *= self.novelty(word)

                    if self._assoc_type == CONST.DEC_SUM:
                        self._aligns.add_decay_sum(word, feature, self._time, alignment, self._forget_decay)
                    else:
                        self._aligns.add_alignment(word, feature, self._time, alignment)

        for word in words:
            self.update_meaning_prob(word)

            if self._novelty_flag or self._grow_graph_flag:
                self._last_time[word] = self._time

            if self._grow_graph_flag and word.endswith(":N") and word not in self._stopwords:
                word_acq_score = self.calculate_acquisition_score(word)
                self._words_graph.add_word(
                    self._context, word, word_acq_score, self._learned_lexicon,
                    self._last_time, self._time, self._beta, self._simtype
                )
                self._context.append(word)
                if len(self._context) > 100:
                    self._context = self._context[1:]
    def calculate_category_probs(self, words, features, category_learner):
        """
        Assign categories to each word, then calculate the probability of each
        word-feature pair based on the category's meaning representation.
        Returns a dictionary of dictionaries: {word: {feature: probability}}.
        """
        beta = self._beta
        simtype = self._simtype
        categories = {}

        for word in words:
            category = category_learner.word_category(word)
            if category == -1:
                wn_category = self._wnlabels.wordnet_label(word)
                category = category_learner.categorize(word, simtype, beta, None, wn_category)
            categories[word] = category

        category_word_feature_probs = {}

        for feature in features:
            for word in words:
                if word not in category_word_feature_probs:
                    category_word_feature_probs[word] = {}

                category = categories[word]
                prob = 1.0 / beta if category == -1 else category_learner.prob(category, feature)
                category_word_feature_probs[word][feature] = prob

        return category_word_feature_probs

    def process_pair(self, words, features, outdir, category_learner=None):
        """
        Process a single word-feature input pair for learning.

        For FAS (alignment_method 0), `features` is a flat list of features
        in the scene. For other models, `features` is a list of referents, where
        each referent is a list of features.
        """
        self._time += 1  # Advance timestep

        for feature in set(flatten(features)):
            self._features.add(feature)

        if self._dummy:
            words.append("dummy")

        if self.alignment_method == 0:
            features = flatten(features)

        self.calculate_alignments(words, features, outdir, category_learner)

        if self._dummy:
            words.remove("dummy")

        if self._stats_flag:
            t = self._time

            for word in words:
                if not self._wordsp.has_word(word):
                    learned_c = self.learned_count(self._postags)
                    self._wordsp.add_word(word, t, learned_c)
                else:
                    self._wordsp.inc_frequency(word)

                self._wordsp.update_last_time(word, t)

                acq = self.calculate_acquisition_score(word)

                if word not in self._acq_score_list:
                    self._acq_score_list[word] = {}
                self._acq_score_list[word][t] = acq

                if word not in self._vocab and acq >= self._theta:
                    lrnd_count = self.learned_count(self._postags)
                    frequency = self._wordsp.frequency(word)

                    self._wordsp.update_lrnd_props(word, t, frequency, lrnd_count)

                    if frequency > self._minfreq:
                        self._vocab.add(word)

    def process_corpus(self, corpus_path, outdir, category_learner=None, corpus=None):
        """
        Process the corpus at `corpus_path`, or the provided `corpus` object.
        Saves any gathered statistics to `outdir`. If a CategoryLearner is provided,
        categories are updated periodically. Returns (num_learned_words, num_timesteps).
        """
        close_corpus = False
        if corpus is None:
            if not os.path.exists(corpus_path):
                print(f"Error -- Corpus does not exist: {corpus_path}")
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True

        words, features = corpus.next_pair()
        learned = 0

        while words:
            if self._maxtime > 0 and self._time >= self._maxtime:
                break

            if self._remove_singletons and len(words) == 1:
                words, features = corpus.next_pair()
                continue

            # Cluster categories periodically
            if (
                self._time > 100 and self._time % 1000 == 0
                and self._category_flag and self._tasktype is not None
            ):
                print("making categories")
                seen_words = self._wordsp._words
                lexicon = self._learned_lexicon
                stopwords = self._stopwords
                all_features = self._all_features
                seen_features = self._features

                clusters, labels, cwords = semantic_clustering_categories(
                    self._beta, seen_words, lexicon, all_features,
                    self._wnlabels, stopwords, CONST.N, 20
                )
                category_learner = CategoryLearner(self._beta, clusters, lexicon, seen_features)

            # Track novel nouns
            noun_count = 0
            novel_nouns = []
            if self._stats_flag:
                for word in words:
                    if word.endswith(":N") and word not in self._stopwords:
                        noun_count += 1
                        if word not in self._wordsp.all_words(0):
                            novel_nouns.append(word)

            self.process_pair(words, features, outdir, category_learner)
            learned = len(self._vocab)

            if self._maxlearned > 0 and learned > self._maxlearned:
                break

            if self._time % 100 == 0:
                print(self._time)

            if self._stats_flag:
                self.record_statistics(corpus_path, words, novel_nouns, noun_count, outdir)

            if self._context_stats_flag:
                sims = self.calculate_similarity_scores(words)
                comps = self.calculate_comprehension_scores(words)
                self._contextsp.add_context(set(words), self._time, sims, comps)

            words, features = corpus.next_pair()

        if self._stats_flag:
            self._wordsp.write(corpus_path, outdir, str(self._time))
            self._timesp.write(corpus_path, outdir, str(self._time))

        if self._context_stats_flag:
            all_words = list(self._contextsp._words.keys())
            sims = self.calculate_similarity_scores(all_words)
            comps = self.calculate_comprehension_scores(all_words)
            for word in all_words:
                self._contextsp.add_similarity(word, sims[word])
                self._contextsp.add_comprehension(word, comps[word])
            self._contextsp.write(corpus_path, outdir)

        if close_corpus:
            corpus.close()

        return learned, self._time

    def record_statistics(self, corpus, words, novel_nouns, noun_count, outdir):
        """
        Record statistics in this learner's TimePropsTable and WordPropsTable
        based on the current time step. Includes general acquisition scores and
        novelty-based scores for nouns.
        """
        if self._record_itrs > 0 and self._time % self._record_itrs == 0:
            self._wordsp.write(corpus, outdir, str(self._time))
            self._timesp.write(corpus, outdir, str(self._time))

        avg_acq = {}

        # Novel noun tracking
        avg_acq_nn = self.avg_acquisition(novel_nouns, CONST.N)
        if noun_count >= 2 and len(novel_nouns) >= 1:
            avg_acq[CONST.NOV_N_MIN1] = avg_acq_nn
        if noun_count >= 2 and len(novel_nouns) >= 2:
            avg_acq[CONST.NOV_N_MIN2] = avg_acq_nn

        all_words = self._wordsp.all_words(self._minfreq)
        all_learned = list(self._vocab)

        avg_acq[CONST.LRN] = self.avg_acquisition(all_learned, CONST.ALL)

        if CONST.ALL in self._postags:
            avg_acq[CONST.ALL] = self.avg_acquisition(all_words, CONST.ALL)
        if CONST.N in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.N] = self.avg_acquisition(all_words, CONST.N)
        if CONST.V in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.V] = self.avg_acquisition(all_words, CONST.V)
        if CONST.OTH in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.OTH] = self.avg_acquisition(all_words, CONST.OTH)

        heard = self.heard_count(self._minfreq, self._postags)
        learned = self.learned_count(self._postags)

        self._timesp.add_time(self._time, heard, learned, avg_acq)

    def calculate_similarity_scores(self, words):
        """
        Calculate the similarity score for each word in the list `words` and
        return a dictionary mapping each word to its cosine similarity score
        between learned and gold meanings.
        """
        sims = {}
        beta = self._beta
        lexicon = self._learned_lexicon
        gold_lexicon = self._gold_lexicon

        for word in words:
            lrnd_m = lexicon.meaning(word)
            true_m = gold_lexicon.meaning(word)
            sims[word] = evaluate.sim_cosine(beta, lrnd_m, true_m)

        return sims

    def calculate_comprehension_scores(self, words):
        """
        Calculate the comprehension score for each word in the list `words` and
        return a dictionary mapping each word to its comprehension score.
        The score is the sum of the probabilities assigned to the correct features.
        """
        comps = {}
        lexicon = self._learned_lexicon
        gold_lexicon = self._gold_lexicon

        for word in words:
            lrnd_m = lexicon.meaning(word)
            true_features = gold_lexicon.meaning(word).seen_features()
            comp = sum(lrnd_m.prob(feature) for feature in true_features)
            comps[word] = comp

        return comps

    def heard_count(self, minfreq, postags):
        """
        Return the number of different words that have been encountered and have
        occurred at least `minfreq` times, with POS tags matching `postags`.
        """
        if CONST.ALL in postags:
            return self._wordsp.count(minfreq)

        words = self._wordsp.all_words(minfreq)
        return postag_count(words, postags)

    def learned_count(self, postags):
        """
        Return the number of learned words whose part-of-speech tags match `postags`.
        """
        if CONST.ALL in postags:
            return len(self._vocab)

        vocab_list = list(self._vocab)
        return postag_count(vocab_list, postags)







