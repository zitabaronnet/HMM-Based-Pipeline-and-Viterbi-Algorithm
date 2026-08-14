import os
from pathlib import Path
from typing import List, Union, Callable, ClassVar, Optional, Set, Tuple, Iterator
import numpy as np

from novelshare.hash import hash_tokens
import re

from collections import Counter, defaultdict
from itertools import product

import pandas as pd

from decimal import *
getcontext().prec = 100


def parse_data(path: Path):

    assert path.exists()

    tokens_list = []
    annotations_list = []
    with path.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip() == "" or line.startswith("#"):  # New sentence or end of file
                continue

            l_split = line.strip().split()
            annotation = l_split[-1]

            token = l_split[:-1]
            tokens_list.append(token)
            annotations_list.append(annotation)

    return tokens_list, annotations_list


class TokenList:

    hash_len: ClassVar = 2

    def __init__(self, token_list: List[str], annotation_list: List[str] = None, order: int = 1) -> None:
        self.unclean_tokens = token_list
        self.tokens = []
        self.hashed_tokens = []
        self.unclean_annotations = annotation_list
        self.annotations = []
        self.clean_pattern = re.compile(r"[^\w\s]+")
        self.first_word = None
        self.word_counter = None
        self.cleaned_tokens = False
        self.order = order

    @staticmethod
    def from_path(path: Union[Path, str], order: int = 1) -> 'TokenList':

        if isinstance(path, str):
            path = Path(path)

        tok, an = parse_data(path)
        return TokenList(tok, an, order=order)

    def clean_and_tokenize(self) -> None:

        for idx, word_list in enumerate(self.unclean_tokens):
            for word in word_list:
                cleaned_word_lst = self.clean_pattern.split(word)
                for cleaned_word in cleaned_word_lst:
                    if cleaned_word != "":
                        self.tokens.append(cleaned_word)
                        self.annotations.append(self.unclean_annotations[idx])

        self.first_word = self.tokens[0] if  self.order == 1 else tuple(self.tokens[:self.order])

        self.word_counter = Counter(self.tokens)
        self.hashed_tokens = hash_tokens(self.tokens, hash_len=self.hash_len)
        self.cleaned_tokens = True

    def get_tokens(self) -> List[str]:

        if not self.cleaned_tokens:
            print("Must call clean_and_tokenize_first")
            return []

        return self.tokens

    def get_annotations(self) -> List[str]:

        if not self.cleaned_tokens:
            print("Must call clean_and_tokenize_first")
            return []

        return self.annotations

    def get_word_counter(self) -> Counter:

        if not self.cleaned_tokens:
            print("Must call clean_and_tokenize_first")
            return []

        return self.word_counter

    def get_first_word(self) -> str:

        if not self.cleaned_tokens:
            print("Must call clean_and_tokenize_first")
            return []

        return self.first_word

    def get_hashed_tokens(self) -> List[str]:

        if not self.cleaned_tokens:
            print("Must call clean_and_tokenize_first")
            return []

        return self.hashed_tokens


class TokenHMMData:

    def __init__(self, token_lists: List[TokenList], additional_tokens: List[str] = None, order: int = 1) -> None:

        assert len(token_lists) > 0

        self.token_lists = token_lists
        self.order = order

        total_word_counter = token_lists[0].get_word_counter()
        unique_tokens = set(token_lists[0].get_tokens())
        first_word_list = [token_lists[0].get_first_word()]

        for lst in self.token_lists[1:]:
            total_word_counter = total_word_counter + lst.get_word_counter()
            unique_tokens = unique_tokens.union(lst.get_tokens())
            first_word_list.append(lst.get_first_word())

        if additional_tokens is not None:
            unique_tokens = unique_tokens.union(set(additional_tokens))

        self.total_word_counter = total_word_counter
        self.unique_tokens = list(unique_tokens)
        self.total_words = len(total_word_counter)
        self.first_word_counter = Counter(first_word_list)
        self.precomputation_occurred = False
        self.transition_probability_map = defaultdict(lambda : Decimal(0))
        self.preimage_map = defaultdict(lambda : list())
        self.image_map = defaultdict(lambda : list())
        self.hash_preimage_map = defaultdict(lambda : list())
        self.single_hash_to_token_map = defaultdict(lambda : list())
        self.states = self.unique_tokens if self.order == 1 else product(self.unique_tokens,repeat=self.order)
        self.probability_vector = pd.Series(index=self.states, dtype=object)
        self.probability_vector.iloc[:] = Decimal(0)


    @staticmethod
    def from_paths(first_path: Union[str, Path], *args, additional_tokens: List[str] = None, order: int = 1) -> 'TokenHMMData':
        first_list = TokenList.from_path(first_path, order=order)
        first_list.clean_and_tokenize()
        token_lists = [first_list]

        for path in args:
            tok_list = TokenList.from_path(path, order=order)
            tok_list.clean_and_tokenize()
            token_lists.append(tok_list)

        return TokenHMMData(token_lists, additional_tokens=additional_tokens, order=order)

    def precompute_maps(self):


        for v in self.unique_tokens:
            h_v = hash_tokens([v], hash_len=TokenList.hash_len)
            if v not in self.single_hash_to_token_map[h_v[0]]:
                self.single_hash_to_token_map[h_v[0]].append(v)

        total_pairs = 0
        for token_list in self.token_lists:

            token_values = token_list.get_tokens()
            state_sequence = self.order_transform(token_values)

            for state_head, state_tail in zip(state_sequence[:-2], state_sequence[1:]):
                total_pairs += 1

                self.transition_probability_map[(state_head, state_tail)] += Decimal(1)

                if state_head not in self.preimage_map[state_tail]:
                    self.preimage_map[state_tail].append(state_head)

                if state_tail not in self.image_map[state_head]:
                    self.image_map[state_head].append(state_tail)

        # note that since pairs are counted via a sliding window of width 2, the number of pairs equals the number of words
        total_words = total_pairs
        for key, value in self.transition_probability_map.items():

            p_a_and_b = value / Decimal(total_pairs)
            p_b = Decimal(self.total_word_counter[key[0]]) / Decimal(total_words)
            self.transition_probability_map[key] = p_a_and_b / p_b

        """
        if self.order == 1:

            for state in self.states:
                hashed = hash_tokens([state], hash_len=TokenList.hash_len)
                self.hash_preimage_map[hashed[0]].append(state)

        else:

            for state in self.states:
                hashed = hash_tokens(list(state), hash_len=TokenList.hash_len)
                self.hash_preimage_map[tuple(hashed)].append(state)
        """
        self.precomputation_occurred = True

    def order_transform(self, sequence: List[str]) -> Union[List[str], List[Tuple[str, ...]]]:

        if self.order == 1:
            return sequence

        start_offset = 0
        end_offset = -self.order

        sub_sequences = []
        for i in range(self.order):
            sub_sequences.append(sequence[start_offset:end_offset])
            start_offset += 1
            end_offset += 1

        return list(zip(*sub_sequences))


    def get_transition_probabilities(self, start: Union[str, Tuple[str, ...]], end:  Union[str, Tuple[str, ...]]) -> Optional[Decimal]:

        if not self.precomputation_occurred:
            print("Must call precompute_maps first")
            return None

        return self.transition_probability_map[(start, end)]

    def get_transition_prob_vector_for_target(self, current_state: Union[str, Tuple[str, ...]]) -> pd.Series:

        # determines to what extent states with transition probability 0 should be
        # able to manifest in the end result
        #
        DECISION_FACTOR = Decimal(99) / 100
        DECISION_FACTOR_INVERSE = 1 - DECISION_FACTOR

        preimages = self.preimage_map[current_state]
        preimage_vals = [DECISION_FACTOR * self.transition_probability_map[pre, current_state] for pre in preimages]
        remaining_states = len(self.states) - len(preimages)
        leftover_prob = DECISION_FACTOR_INVERSE / remaining_states
        self.probability_vector.iloc[:] = leftover_prob

        self.probability_vector.loc[preimages] = preimage_vals
        return self.probability_vector

    def get_initial_probability(self, token:  Union[str, Tuple[str, ...]]) -> Decimal:
        return Decimal(self.first_word_counter.get(token, 0)) / Decimal(self.total_words)


    def get_hash_preimage(self, hash: Union[str, Tuple[str, ...]]) -> Optional[Iterator[Union[str, Tuple[str, ...]]]]:

        if not self.precomputation_occurred:
            print("Must call precompute_maps first")
            return None

        if self.order == 1:
            return self.single_hash_to_token_map[hash]
        else:
            init_iter = [self.single_hash_to_token_map[hash[0]]]
            for i in range(1, len(hash)):
                init_iter.append(self.single_hash_to_token_map[hash[i]])
            return product(*init_iter)

    def get_previous_state_iter(self, current_observation: Union[str, List[str]]):
        if self.order == 1:
            return self.unique_tokens
        else:
            # in the higher order case we know that hashes / emitted symbols overlap
            # e.g. previous_token = ('A1', 'B2'), current_token = ('B2','3F')
            # i.e. the first self.order hashes are the last self.order hashes of the previous one
            first_iter = [self.unique_tokens]
            for hash_i in current_observation[:self.order-1]:
                first_iter.append(self.single_hash_to_token_map[hash_i])
            return product(*first_iter)



    def get_possible_image(self, token:  Union[str, Tuple[str, ...]]) -> List[ Union[str, Tuple[str, ...]]]:
        return self.image_map[token]

    def get_possible_preimage(self, token:  Union[str, Tuple[str, ...]]) -> List[ Union[str, Tuple[str, ...]]]:
        return self.preimage_map[token]


    def get_unique_tokens(self) -> List[str]:
        return self.unique_tokens

    def get_states(self) -> List[ Union[str, Tuple[str, ...]]]:
        return list(self.states)

    def get_token_list(self, index: int) -> TokenList:
        return self.token_lists[index]


def viterbi(observed_seq, hmm_data: TokenHMMData):

    observed_seq = hmm_data.order_transform(observed_seq)

    states = hmm_data.get_states()

    n_states = len(states)
    n_obs = len(observed_seq)

    # probability_table = [[Decimal(0)] * n_states] * n_obs
    probability_df = pd.DataFrame(index=list(range(n_obs)), columns=states,dtype=object)
    probability_df.fillna(Decimal('0'), inplace=True)

    fill_lambda = lambda : "" if hmm_data.order == 1 else tuple([""] * hmm_data.order)
    fill_data = fill_lambda()

    # previous_state_table = [[fill_lambda()] * n_states] * n_obs
    previous_state_df = pd.DataFrame(index=list(range(n_obs)), columns=states)
    previous_state_df.fillna(fill_data, inplace=True)

    # Note in our case the emission probs are binary, i.e. given a state the probability to observe an emission is either 0 or 1
    # so we only need to go over the non-zero probs

    for state_token in hmm_data.get_hash_preimage(observed_seq[0]):
        probability_df.loc[0, [state_token]] = hmm_data.get_initial_probability(state_token)

    for timepoint in range(1, n_obs):
        #print(timepoint, end=", ")
        oktp = False
        # s
        for current_token in hmm_data.get_hash_preimage(observed_seq[timepoint]):

            p_vec = hmm_data.get_transition_prob_vector_for_target(current_token)
            new_prov_vec = probability_df.loc[timepoint - 1, :] * p_vec
            maximum_index = new_prov_vec.argmax()
            maximum_prob = new_prov_vec.iloc[maximum_index]
            maximal_previous_state = new_prov_vec.index[maximum_index]
            probability_df.loc[timepoint, [current_token]] = maximum_prob
            previous_state_df.loc[timepoint, [current_token]] = [maximal_previous_state]

            if maximum_prob > Decimal(0):
                oktp = True

        assert oktp

    #print("Reconstru")
    reconstructed_states = [""] * n_obs
    reconstructed_states[-1] = states[probability_df.loc[n_obs - 1, :].argmax()]

    for timepoint in range(2, n_obs + 1):
        last_state = reconstructed_states[-timepoint+1]
        #print(f"previous_state_df.loc[{n_obs - timepoint}, {last_state}]")
        current_state = previous_state_df.loc[n_obs - timepoint + 1][last_state]
        #print(timepoint, reconstructed_states, current_state)
        reconstructed_states[-timepoint] = current_state

    # stitch solution back
    stitched_states = []
    if hmm_data.order > 1:
        for v in reconstructed_states:
            stitched_states.append(v[0])
        reconstructed_states = stitched_states

    return reconstructed_states, probability_df, previous_state_df

def measure_error_rate(result: List[str], expected: List[str]) -> float:

    ctr = 0
    for a, b in zip(result, expected):
        if a != b:
            ctr += 1
    return ctr / len(result)

def compare_sequence(expected_output: List[str], predicted_output: List[str], hash_sequence: List[str],  hmm_data: TokenHMMData):

    preimage_count_cor = []
    preimage_count_inc = []

    mistakes = 0
    for a_tok, e_tok, hash_  in zip(predicted_output, expected_output, hash_sequence):
        pre = hmm_data.get_hash_preimage(hash_)
        if a_tok == e_tok:
            preimage_count_cor.append(len(pre))
        else:
            mistakes += 1
            preimage_count_inc.append(len(pre))

    misprediction_rate = mistakes / len(hash_sequence)
    m_cor, med_cor, var_cor = np.mean(preimage_count_cor), np.median(preimage_count_cor), np.var(preimage_count_cor)
    m_inc, med_inc, var_inc, = np.mean(preimage_count_inc), np.median(preimage_count_inc), np.var(preimage_count_inc)

    return misprediction_rate, (m_cor, med_cor, var_cor), (m_inc, med_inc, var_inc)


def plot_hash_distribution(hmm: TokenHMMData):

    labels = []
    buckets = []
    for h in product("1234567890abcdef",repeat=2):
        label = "".join(h)
        labels.append(h)
        buckets.append((label, len(hmm.get_hash_preimage(label))))

    bs = sorted(buckets, key=lambda x: -x[1])
    bs_slice = bs[:20]

    bs_labels = [a for a, b in bs_slice]
    bs_values = [b for a, b in bs_slice]

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('darkgrid')
    sns.set_palette("colorblind")


    pp = sns.histplot(x=bs_labels,y=bs_values,bins=len(bs_labels))
    pp.set_xlabel("Truncated Hash Values", fontsize=15)
    pp.set_ylabel("Amount of preimages", fontsize=15)
    plt.tight_layout()
    plt.savefig("./hash_freq_most.eps")
    pass

def run_single_viterbi_test(paths_train: List[Union[str, Path]], path_test: Union[str, Path], cutoff=100):

    paths_train = [p for p in paths_train if p.name != path_test.name]

    corpus_test = TokenHMMData.from_paths(path_test)
    corpus_test.precompute_maps()

    corpus_train = TokenHMMData.from_paths(*paths_train, additional_tokens=corpus_test.get_unique_tokens())
    corpus_train.precompute_maps()

    # plot_hash_distribution(corpus_train)

    sample_list = corpus_test.get_token_list(0)
    sample_full = sample_list.get_hashed_tokens()
    sample = sample_full[:cutoff]
    expected_full = sample_list.get_tokens()
    expected = expected_full[:cutoff]

    predicted_tokens, _, _ = viterbi(sample, corpus_train)

    statistics = compare_sequence(expected, predicted_tokens, sample, corpus_train)

    return statistics


def load_all_corpora(folders: List[str]):
    all_paths = []
    for folder in folders:
        pp = Path(folder).glob("*.conll")
        all_paths.extend(pp)
    return all_paths

def run_viterbi_loo(folder_path: str, cutoff=-1):
    from os import listdir
    all_docs = load_all_corpora([folder_path+p for p in listdir(folder_path) if not p.startswith(".")] )

    if cutoff == -1:
        cutoff = len(all_docs)

    results = []
    for i in range(cutoff):
        test_data = all_docs[i]
        train_data = all_docs[:i] + all_docs[(i+1):]
        try:
            rat, cor_stat, inc_stat = run_single_viterbi_test(train_data, test_data)
            results.append((rat, i))
            print(rat)
        except Exception as _:
            pass
        continue

    total_results = [a for a, b in results]

    return print("[Alignment] LOO misprediction rate = ", np.mean(total_results), "median misprediction rate", np.median(total_results))



def run_single_alignment(paths_train: List[Union[str, Path]], path_test: Union[str, Path], cutoff=100):

    from novelshare.align import align_tokens, make_plugin_case, make_plugin_propagate, make_plugin_retokenize, make_plugin_mlm
    assert path_test not in paths_train

    paths_train = [p for p in paths_train if p.name != path_test.name]
    corpus_test = TokenHMMData.from_paths(path_test)
    corpus_test.precompute_maps()

    corpus_train = TokenHMMData.from_paths(*paths_train, additional_tokens=corpus_test.get_unique_tokens())
    corpus_train.precompute_maps()

    sample_list = corpus_test.get_token_list(0)
    sample_full = sample_list.get_hashed_tokens()
    to_predict = sample_full[:cutoff]
    expected_full = sample_list.get_tokens()
    expected = expected_full[:cutoff]


    all_scores = []

    for i in range(len(paths_train)):
        doc_token_list = corpus_train.get_token_list(i)
        sample_tokens = doc_token_list.get_tokens()

        sample_tokens_cut = sample_tokens[:cutoff]

        aligned = align_tokens(to_predict, sample_tokens_cut,hash_len=TokenList.hash_len,alignment_plugins=[make_plugin_retokenize(max_token_len=24, max_splits_nb=4),
                make_plugin_case(),
                make_plugin_propagate()])

        score = 0
        for a,b in zip(expected, aligned):
            if a != b:
                score += 1
        all_scores.append((score / len(expected), i))

    final = min(all_scores, key=lambda x: x[0])
    return final[0]


def run_alignment_loo(folder_path: str, cutoff=-1):
    from os import listdir
    all_docs = load_all_corpora([folder_path+p for p in listdir(folder_path) if not p.startswith(".")] )

    if cutoff == -1:
        cutoff = len(all_docs)

    total_results = []
    for i in range(cutoff):
        test_data = all_docs[i]
        train_data = all_docs[:i] + all_docs[(i+1):]
        rat = run_single_alignment(train_data, test_data)
        total_results.append(rat)
    print("[Alignment] LOO misprediction rate = ", np.mean(total_results), "median misprediction rate", np.median(total_results))



if __name__ == "__main__":

    run_viterbi_loo("data/Moby_Dick/", cutoff=10)
    run_alignment_loo("data/Pride_and_Prejudice/")
