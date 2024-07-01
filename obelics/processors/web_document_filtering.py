import json
import re
from collections import Counter
from tqdm import tqdm

import fasttext
import numpy as np

from obelics.utils import SPECIAL_CHARACTERS


class FilteringFunctions:

    @staticmethod
    def remove_empty_el_from_list(list_):
        return [el for el in list_ if el]

    @staticmethod
    def remove_non_printing_characters(text, non_printing_characters_re):
        return non_printing_characters_re.sub("", text)

    @staticmethod
    def standardize_whitespace(
        text,
        whitespace=[
            " ",
            " ",
            " ",
            " ",
            " ",
            "　",
            " ",
            " ",
            " ",
            " ",
            "￼",
            "",
            " ",
        ],
    ):
        """There are different whitespace characters."""
        whitespace = set(whitespace)
        text = "".join([char if char not in whitespace else " " for char in text])
        return text

    @staticmethod
    def split_on_whitespace(
        text,
        new_line=False,
        tab=False,
    ):
        """This method also removes concatenated spaces."""
        sep = [" "] + new_line * ["\n"] + tab * ["\t"]
        sep = "|".join(sep)
        split_text = re.split(sep, text)
        split_text = FilteringFunctions.remove_empty_el_from_list(split_text)
        return split_text

    @staticmethod
    def strip(text, strip_characters):
        """Way faster than text.strip(strip_characters)
        since strip_characters is a set instead of a str,
        and it contains a lot of elements (all the emojis)."""
        if not text:
            return text
        beg_ind = 0
        end_ind = len(text)
        for i in range(len(text)):
            if text[i] in strip_characters:
                beg_ind += 1
            else:
                break
        for i in range(1, len(text) + 1):
            if text[-i] in strip_characters:
                end_ind -= 1
            else:
                break
        text_stripped = text[beg_ind:end_ind]
        return text_stripped

    @staticmethod
    def get_words_from_text(text, lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS):
        """Get words from a text. Non reversible since the text
        is split on multiple characters, words are stripped of
        special characters and characters are converted to lower case.
        Useful to compute ratios, like the stopword ratio."""
        if strip_words and strip_characters is None:
            raise ValueError("strip_characters must be provided if strip_words is True.")
        words = FilteringFunctions.split_on_whitespace(text=text, new_line=True, tab=True)
        if lower_case:
            words = [word.lower() for word in words]
        if strip_words:
            words = [FilteringFunctions.strip(word, strip_characters) for word in words]
            words = FilteringFunctions.remove_empty_el_from_list(words)
        return words

    @staticmethod
    def check_number_words(text, strip_characters, number_words_min_cutoff, number_words_max_cutoff):
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        number_words = len(words)
        if (number_words < number_words_min_cutoff) or (number_words > number_words_max_cutoff):
            return False
        return True

    @staticmethod
    def compute_character_repetition_ratio(text, character_repetition_length):
        def get_freq_character_ngrams(text, n):
            character_ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
            freq_character_ngrams = Counter(character_ngrams)
            return freq_character_ngrams

        freq_character_ngrams = get_freq_character_ngrams(text=text, n=character_repetition_length)
        if len(freq_character_ngrams) == 0:
            return 0
        freq_character_ngrams = list(freq_character_ngrams.values())
        freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
        val_one = len([el for el in freq_character_ngrams if el == 1])
        num_rep_character_ngrams = min(
            int(np.sqrt(len(freq_character_ngrams))),
            len(freq_character_ngrams) - val_one,
        )
        character_repetition_ratio = sum(freq_character_ngrams[:num_rep_character_ngrams]) / sum(freq_character_ngrams)
        return character_repetition_ratio

    @staticmethod
    def check_character_repetition_ratio(
        text,
        character_repetition_length,
        character_repetition_max_cutoff,
    ):
        character_repetition_ratio = FilteringFunctions.compute_character_repetition_ratio(
            text=text, character_repetition_length=character_repetition_length
        )
        if character_repetition_ratio > character_repetition_max_cutoff:
            return False
        return True

    @staticmethod
    def compute_word_repetition_ratio(text, strip_characters, word_repetition_length):
        def get_freq_word_ngrams(text, strip_characters, n):
            words = FilteringFunctions.get_words_from_text(
                text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
            )
            word_ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            freq_word_ngrams = Counter(word_ngrams)
            return freq_word_ngrams

        freq_word_ngrams = get_freq_word_ngrams(text=text, strip_characters=strip_characters, n=word_repetition_length)
        if len(freq_word_ngrams) == 0:
            return 0
        freq_word_ngrams = list(freq_word_ngrams.values())
        word_repetition_ratio = sum(freq for freq in freq_word_ngrams if freq > 1) / sum(freq_word_ngrams)
        return word_repetition_ratio

    @staticmethod
    def check_word_repetition_ratio(
        text,
        strip_characters,
        word_repetition_length,
        word_repetition_max_cutoff,
    ):
        word_repetition_ratio = FilteringFunctions.compute_word_repetition_ratio(
            text=text, strip_characters=strip_characters, word_repetition_length=word_repetition_length
        )
        cond = word_repetition_ratio <= word_repetition_max_cutoff
        return cond

    @staticmethod
    def compute_special_character_ratio(text, special_characters):
        if len(text) == 0:
            return 0
        special_character_ratio = len([char for char in text if char in special_characters]) / len(text)
        return special_character_ratio

    @staticmethod
    def check_special_character_ratio(text, special_characters, special_character_ratio_max_cutoff):
        special_character_ratio = FilteringFunctions.compute_special_character_ratio(
            text=text, special_characters=special_characters
        )
        if special_character_ratio > special_character_ratio_max_cutoff:
            return False
        return True

    @staticmethod
    # def compute_stopword_ratio(text, strip_characters, stopwords):
    #     words = FilteringFunctions.get_words_from_text(
    #         text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
    #     )
    #     if not words:
    #         return 0
    #     stopword_ratio = len([word for word in words if word in stopwords]) / len(words)
    #     return stopword_ratio
    #
    # @staticmethod
    # def check_stopword_ratio(text, strip_characters, stopwords, stopword_ratio_min_cutoff):
    #     stopword_ratio = FilteringFunctions.compute_stopword_ratio(
    #         text=text, strip_characters=strip_characters, stopwords=stopwords
    #     )
    #     if stopword_ratio < stopword_ratio_min_cutoff:
    #         return False
    #     return True

    @staticmethod
    def compute_flagged_word_ratio(text, strip_characters, flagged_words):
        words = FilteringFunctions.get_words_from_text(
            text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
        )
        if not words:
            return 0
        flagged_word_ratio = len([word for word in words if word in flagged_words]) / len(words)
        return flagged_word_ratio

    @staticmethod
    def check_flagged_word_ratio(
        text,
        strip_characters,
        flagged_words,
        flagged_word_ratio_max_cutoff,
    ):
        flagged_word_ratio = FilteringFunctions.compute_flagged_word_ratio(
            text=text,
            strip_characters=strip_characters,
            flagged_words=flagged_words,
        )
        if flagged_word_ratio > flagged_word_ratio_max_cutoff:
            return False
        return True

    # @staticmethod
    # def compute_punctuation_ratio(text, punctuation, min_nb_words=-1):
    #     # The regepx is used to extract all words and punctuation marks from the text string.
    #     # `[\w']+`: This matches any sequence of one or more word characters (letters, digits, or underscores) or apostrophes.
    #     # The square brackets indicate a character class, which means "match any of these characters".
    #     # The backslash before the w means to escape it as it has a special meaning in regex,
    #     # and the plus sign means "one or more occurrences". We choose to ignore other special characters because in practise,
    #     # it's a good enough approximation to filter web documents that don't contain long sequences of (English) text.
    #     # `|`: This is the "or" operator in regex. It separates the two alternative patterns that the regex can match.
    #     # `[punctuation]`: This matches any one of the punctuation marks specified inside the square brackets.
    #     # This is inspired from https://stackoverflow.com/a/367292/6226208
    #     punc_splitters = "".join(punctuation)
    #     pattern = r"{}".format(f"[\w']+|[{punc_splitters}]")
    #     punctuation_splitted_words = re.findall(pattern, text)
    #     if not punctuation_splitted_words:
    #         return 0
    #     if min_nb_words > 0:
    #         if len(punctuation_splitted_words) < min_nb_words:
    #             # If the sequence is too short, we don't want to rely on punctuation to filter it out.
    #             return 1.0
    #     punctuation_ratio = len([word for word in punctuation_splitted_words if word in punctuation]) / len(
    #         punctuation_splitted_words
    #     )
    #     return punctuation_ratio
    #
    # @staticmethod
    # def check_punctuation_ratio(
    #     text,
    #     punctuation,
    #     punctuation_ratio_min_cutoff,
    #     min_nb_words=-1,
    # ):
    #     punctuation_ratio = FilteringFunctions.compute_punctuation_ratio(
    #         text=text,
    #         punctuation=punctuation,
    #         min_nb_words=min_nb_words,
    #     )
    #     if punctuation_ratio < punctuation_ratio_min_cutoff:
    #         return False
    #     return True
    #
    # @staticmethod
    # def compute_common_word_ratio(text, strip_characters, common_words):
    #     words = FilteringFunctions.get_words_from_text(
    #         text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
    #     )
    #     if not words:
    #         return 0
    #     common_word_ratio = len([word for word in words if word in common_words]) / len(words)
    #     return common_word_ratio
    #
    # @staticmethod
    # def check_common_word_ratio(
    #     text,
    #     strip_characters,
    #     common_words,
    #     common_word_ratio_min_cutoff,
    # ):
    #     common_word_ratio = FilteringFunctions.compute_common_word_ratio(
    #         text=text,
    #         strip_characters=strip_characters,
    #         common_words=common_words,
    #     )
    #     if common_word_ratio < common_word_ratio_min_cutoff:
    #         return False
    #     return True

    @staticmethod
    def compute_lang_id_pred_score(text, lang_id_model):
        text = text.lower().replace("\n", " ")
        tot_pred_lang_id = lang_id_model.predict(text)
        pred_lang_id = tot_pred_lang_id[0][0].replace("__label__", "")
        score_pred_lang_id = tot_pred_lang_id[1][0]
        return pred_lang_id, score_pred_lang_id

    @staticmethod
    def check_lang_id(text, lang_id_model, target_lang_id, lang_id_min_cutoff):
        pred_lang_id, score_pred_lang_id = FilteringFunctions.compute_lang_id_pred_score(
            text=text, lang_id_model=lang_id_model
        )
        if (pred_lang_id != target_lang_id) or (score_pred_lang_id < lang_id_min_cutoff):
            return False
        return True

    @staticmethod
    def replace_digits_with_zeros(text, digits_re):
        return digits_re.sub("0", text)

    @staticmethod
    def replace_unicode_punctuation(text, unicode_punctuation):
        return "".join(unicode_punctuation.get(c, c) for c in text)

    @staticmethod
    def normalization(
        text,
        remove_non_printing_characters,
        strip,
        lower_case,
        standardize_whitespace,
        replace_digits_with_zeros,
        replace_unicode_punctuation,
        non_printing_characters_re,
        digits_re,
        unicode_punctuation,
    ):
        if remove_non_printing_characters:
            text = FilteringFunctions.remove_non_printing_characters(
                text=text, non_printing_characters_re=non_printing_characters_re
            )
        if strip:
            text = text.strip()
        if not text:
            return text
        if lower_case:
            text = text.lower()
        if standardize_whitespace:
            text = FilteringFunctions.standardize_whitespace(text=text)
        if replace_digits_with_zeros:
            text = FilteringFunctions.replace_digits_with_zeros(text=text, digits_re=digits_re)
        if replace_unicode_punctuation:
            text = FilteringFunctions.replace_unicode_punctuation(text=text, unicode_punctuation=unicode_punctuation)
        return text

    @staticmethod
    def tokenization(text, sentencepiece_model, join_on_whitespace):
        text_tokenized = sentencepiece_model.encode_as_pieces(text)
        if join_on_whitespace:
            text_tokenized = " ".join(text_tokenized)
        return text_tokenized


class WebDocumentFilteringNodeLevel:
    __slots__ = (
        "cond_remove_non_printing_characters",
        "non_printing_characters_re",
        "cond_standardize_whitespace",
        "cond_check_number_words_node_level",
        "strip_characters",
        "number_words_node_level_min_cutoff",
        "number_words_node_level_max_cutoff",
        "cond_check_character_repetition_ratio_node_level",
        "character_repetition_length_node_level",
        "character_repetition_node_level_max_cutoff",
        "cond_check_word_repetition_ratio_node_level",
        "word_repetition_length_node_level",
        "word_repetition_node_level_max_cutoff",
        "cond_check_special_character_ratio_node_level",
        "special_character_ratio_node_level_max_cutoff",
        # "cond_check_stopword_ratio_node_level",
        # "stopwords",
        # "stopword_ratio_node_level_min_cutoff",
        "cond_check_flagged_word_ratio_node_level",
        "flagged_words",
        "flagged_word_ratio_node_level_max_cutoff",
        # "cond_check_punctuation_ratio_node_level",
        # "min_number_words_to_check_punctuation_ratio_node_level",
        # "punctuation",
        # "punctuation_ratio_node_level_min_cutoff",
        # "cond_check_common_word_ratio_node_level",
        # "path_common_words",
        # "common_words",
        # "common_word_ratio_node_level_min_cutoff",
        "cond_check_lang_id_node_level",
        "path_lang_id_model",
        "lang_id_model",
        "lang_id_node_level_min_cutoff",
    )

    def __init__(
        self,
        cond_remove_non_printing_characters,
        non_printing_characters_re,
        cond_standardize_whitespace,
        cond_check_number_words_node_level,
        strip_characters,
        number_words_node_level_min_cutoff,
        number_words_node_level_max_cutoff,
        cond_check_character_repetition_ratio_node_level,
        character_repetition_length_node_level,
        character_repetition_node_level_max_cutoff,
        cond_check_word_repetition_ratio_node_level,
        word_repetition_length_node_level,
        word_repetition_node_level_max_cutoff,
        cond_check_special_character_ratio_node_level,
        special_character_ratio_node_level_max_cutoff,
        # cond_check_stopword_ratio_node_level,
        # stopwords,
        # stopword_ratio_node_level_min_cutoff,
        cond_check_flagged_word_ratio_node_level,
        flagged_words,
        flagged_word_ratio_node_level_max_cutoff,
        # cond_check_punctuation_ratio_node_level,
        # min_number_words_to_check_punctuation_ratio_node_level,
        # punctuation,
        # punctuation_ratio_node_level_min_cutoff,
        # cond_check_common_word_ratio_node_level,
        # path_common_words,
        # common_word_ratio_node_level_min_cutoff,
        cond_check_lang_id_node_level,
        path_lang_id_model,
        lang_id_node_level_min_cutoff,
    ):

        self.cond_remove_non_printing_characters = cond_remove_non_printing_characters
        self.non_printing_characters_re = non_printing_characters_re

        self.cond_standardize_whitespace = cond_standardize_whitespace

        self.cond_check_number_words_node_level = cond_check_number_words_node_level
        self.strip_characters = strip_characters
        self.number_words_node_level_min_cutoff = number_words_node_level_min_cutoff
        self.number_words_node_level_max_cutoff = number_words_node_level_max_cutoff

        self.cond_check_character_repetition_ratio_node_level = cond_check_character_repetition_ratio_node_level
        self.character_repetition_length_node_level = character_repetition_length_node_level
        self.character_repetition_node_level_max_cutoff = character_repetition_node_level_max_cutoff

        self.cond_check_word_repetition_ratio_node_level = cond_check_word_repetition_ratio_node_level
        self.word_repetition_length_node_level = word_repetition_length_node_level
        self.word_repetition_node_level_max_cutoff = word_repetition_node_level_max_cutoff

        self.cond_check_special_character_ratio_node_level = cond_check_special_character_ratio_node_level
        self.special_character_ratio_node_level_max_cutoff = special_character_ratio_node_level_max_cutoff

        # self.cond_check_stopword_ratio_node_level = cond_check_stopword_ratio_node_level
        # self.stopwords = stopwords
        # self.stopword_ratio_node_level_min_cutoff = stopword_ratio_node_level_min_cutoff

        self.cond_check_flagged_word_ratio_node_level = cond_check_flagged_word_ratio_node_level
        self.flagged_words = flagged_words
        self.flagged_word_ratio_node_level_max_cutoff = flagged_word_ratio_node_level_max_cutoff

        # self.cond_check_punctuation_ratio_node_level = cond_check_punctuation_ratio_node_level
        # self.min_number_words_to_check_punctuation_ratio_node_level = (
        #     min_number_words_to_check_punctuation_ratio_node_level
        # )
        # self.punctuation = punctuation
        # self.punctuation_ratio_node_level_min_cutoff = punctuation_ratio_node_level_min_cutoff
        #
        # self.cond_check_common_word_ratio_node_level = cond_check_common_word_ratio_node_level
        # self.path_common_words = path_common_words
        # with open(path_common_words) as f:
        #     self.common_words = json.load(f)
        # self.common_word_ratio_node_level_min_cutoff = common_word_ratio_node_level_min_cutoff

        self.cond_check_lang_id_node_level = cond_check_lang_id_node_level
        self.path_lang_id_model = path_lang_id_model
        if cond_check_lang_id_node_level:
            self.lang_id_model = fasttext.load_model(path_lang_id_model)
        self.lang_id_node_level_min_cutoff = lang_id_node_level_min_cutoff

    def __call__(self, dict_md52text):

        dict_md52text_new = {}
        n_p_total, n_p_remain = 0, 0

        for ind, (md5, info) in tqdm(enumerate(dict_md52text.items()), total=len(dict_md52text)):
            if info['Title'] is None or info['Text'] is None:
                continue

            if (len(info['Title']) < 4) or (len(info['Text']) < 4):
                continue

            text = info['Text']

            if self.cond_remove_non_printing_characters:
                text = FilteringFunctions.remove_non_printing_characters(
                    text=text, non_printing_characters_re=self.non_printing_characters_re
                )

            if self.cond_standardize_whitespace:
                text = FilteringFunctions.standardize_whitespace(text=text)

            paragraphs = text.split("\n")
            paragraphs_indices_to_remove = set()

            for ind_par, paragraph in enumerate(paragraphs):
                if paragraph == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED":
                    continue

                if self.cond_check_number_words_node_level:
                    if not FilteringFunctions.check_number_words(
                        text=paragraph,
                        strip_characters=self.strip_characters,
                        number_words_min_cutoff=self.number_words_node_level_min_cutoff,
                        number_words_max_cutoff=self.number_words_node_level_max_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                if self.cond_check_character_repetition_ratio_node_level:
                    if not FilteringFunctions.check_character_repetition_ratio(
                        text=paragraph,
                        character_repetition_length=self.character_repetition_length_node_level,
                        character_repetition_max_cutoff=self.character_repetition_node_level_max_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                if self.cond_check_word_repetition_ratio_node_level:
                    if not FilteringFunctions.check_word_repetition_ratio(
                        text=paragraph,
                        strip_characters=self.strip_characters,
                        word_repetition_length=self.word_repetition_length_node_level,
                        word_repetition_max_cutoff=self.word_repetition_node_level_max_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                if self.cond_check_special_character_ratio_node_level:
                    if not FilteringFunctions.check_special_character_ratio(
                        text=paragraph,
                        special_characters=self.strip_characters,
                        special_character_ratio_max_cutoff=self.special_character_ratio_node_level_max_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                # if self.cond_check_stopword_ratio_node_level:
                #     if not FilteringFunctions.check_stopword_ratio(
                #         text=paragraph,
                #         strip_characters=self.strip_characters,
                #         stopwords=self.stopwords,
                #         stopword_ratio_min_cutoff=self.stopword_ratio_node_level_min_cutoff,
                #     ):
                #         paragraphs_indices_to_remove.add(ind_par)
                #         continue

                if self.cond_check_flagged_word_ratio_node_level:
                    if not FilteringFunctions.check_flagged_word_ratio(
                        text=paragraph,
                        strip_characters=self.strip_characters,
                        flagged_words=self.flagged_words,
                        flagged_word_ratio_max_cutoff=self.flagged_word_ratio_node_level_max_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                # if self.cond_check_punctuation_ratio_node_level:
                #     if not FilteringFunctions.check_punctuation_ratio(
                #         text=paragraph,
                #         punctuation=self.punctuation,
                #         punctuation_ratio_min_cutoff=self.punctuation_ratio_node_level_min_cutoff,
                #         min_nb_words=self.min_number_words_to_check_punctuation_ratio_node_level,
                #     ):
                #         paragraphs_indices_to_remove.add(ind_par)
                #         continue
                #
                # if self.cond_check_common_word_ratio_node_level:
                #     if not FilteringFunctions.check_common_word_ratio(
                #         text=paragraph,
                #         strip_characters=self.strip_characters,
                #         common_words=self.common_words,
                #         common_word_ratio_min_cutoff=self.common_word_ratio_node_level_min_cutoff,
                #     ):
                #         paragraphs_indices_to_remove.add(ind_par)
                #         continue

                if self.cond_check_lang_id_node_level:
                    if not FilteringFunctions.check_lang_id(
                        text=paragraph,
                        lang_id_model=self.lang_id_model,
                        target_lang_id="en",
                        lang_id_min_cutoff=self.lang_id_node_level_min_cutoff,
                    ):
                        paragraphs_indices_to_remove.add(ind_par)
                        continue

                new_paragraphs = [
                    el for ind_par, el in enumerate(paragraphs) if ind_par not in paragraphs_indices_to_remove
                ]
                if not new_paragraphs:
                    continue
                else:
                    dict_md52text_new[md5] = dict_md52text[md5].copy()
                    dict_md52text_new[md5]['Text'] = "\n".join(new_paragraphs)
                    n_p_total += len(paragraphs)
                    n_p_remain += len(new_paragraphs)

        return dict_md52text_new, n_p_total, n_p_remain

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.cond_remove_non_printing_characters,
                self.non_printing_characters_re,
                self.cond_standardize_whitespace,
                self.cond_check_number_words_node_level,
                self.strip_characters,
                self.number_words_node_level_min_cutoff,
                self.number_words_node_level_max_cutoff,
                self.cond_check_character_repetition_ratio_node_level,
                self.character_repetition_length_node_level,
                self.character_repetition_node_level_max_cutoff,
                self.cond_check_word_repetition_ratio_node_level,
                self.word_repetition_length_node_level,
                self.word_repetition_node_level_max_cutoff,
                self.cond_check_special_character_ratio_node_level,
                self.special_character_ratio_node_level_max_cutoff,
                # self.cond_check_stopword_ratio_node_level,
                # self.stopwords,
                # self.stopword_ratio_node_level_min_cutoff,
                self.cond_check_flagged_word_ratio_node_level,
                self.flagged_words,
                self.flagged_word_ratio_node_level_max_cutoff,
                # self.cond_check_punctuation_ratio_node_level,
                # self.min_number_words_to_check_punctuation_ratio_node_level,
                # self.punctuation,
                # self.punctuation_ratio_node_level_min_cutoff,
                # self.cond_check_common_word_ratio_node_level,
                # self.path_common_words,
                # self.common_word_ratio_node_level_min_cutoff,
                self.cond_check_lang_id_node_level,
                self.path_lang_id_model,
                self.lang_id_node_level_min_cutoff,
            ),
        )


class WebDocumentFilteringDocLevel:
    def __init__(
        self,
        cond_check_number_words_doc_level,
        strip_characters,
        number_words_doc_level_min_cutoff,
        number_words_doc_level_max_cutoff,
        cond_check_character_repetition_ratio_doc_level,
        character_repetition_length_doc_level,
        character_repetition_doc_level_max_cutoff,
        cond_check_word_repetition_ratio_doc_level,
        word_repetition_length_doc_level,
        word_repetition_doc_level_max_cutoff,
        cond_check_special_character_ratio_doc_level,
        special_character_ratio_doc_level_max_cutoff,
        # cond_check_stopword_ratio_doc_level,
        # stopwords,
        # stopword_ratio_doc_level_min_cutoff,
        cond_check_flagged_word_ratio_doc_level,
        flagged_words,
        flagged_word_ratio_doc_level_max_cutoff,
        # cond_check_punctuation_ratio_doc_level,
        # punctuation,
        # punctuation_ratio_doc_level_min_cutoff,
        # cond_check_common_word_ratio_doc_level,
        # path_common_words,
        # common_word_ratio_doc_level_min_cutoff,
        cond_check_lang_id_doc_level,
        path_lang_id_model,
        lang_id_doc_level_min_cutoff,
    ):

        self.cond_check_number_words_doc_level = cond_check_number_words_doc_level
        self.strip_characters = strip_characters
        self.number_words_doc_level_min_cutoff = number_words_doc_level_min_cutoff
        self.number_words_doc_level_max_cutoff = number_words_doc_level_max_cutoff

        self.cond_check_character_repetition_ratio_doc_level = cond_check_character_repetition_ratio_doc_level
        self.character_repetition_length_doc_level = character_repetition_length_doc_level
        self.character_repetition_doc_level_max_cutoff = character_repetition_doc_level_max_cutoff

        self.cond_check_word_repetition_ratio_doc_level = cond_check_word_repetition_ratio_doc_level
        self.word_repetition_length_doc_level = word_repetition_length_doc_level
        self.word_repetition_doc_level_max_cutoff = word_repetition_doc_level_max_cutoff

        self.cond_check_special_character_ratio_doc_level = cond_check_special_character_ratio_doc_level
        self.special_character_ratio_doc_level_max_cutoff = special_character_ratio_doc_level_max_cutoff

        # self.cond_check_stopword_ratio_doc_level = cond_check_stopword_ratio_doc_level
        # self.stopwords = stopwords
        # self.stopword_ratio_doc_level_min_cutoff = stopword_ratio_doc_level_min_cutoff

        self.cond_check_flagged_word_ratio_doc_level = cond_check_flagged_word_ratio_doc_level
        self.flagged_words = flagged_words
        self.flagged_word_ratio_doc_level_max_cutoff = flagged_word_ratio_doc_level_max_cutoff

        # self.cond_check_punctuation_ratio_doc_level = cond_check_punctuation_ratio_doc_level
        # self.punctuation = punctuation
        # self.punctuation_ratio_doc_level_min_cutoff = punctuation_ratio_doc_level_min_cutoff
        #
        # self.cond_check_common_word_ratio_doc_level = cond_check_common_word_ratio_doc_level
        # self.path_common_words = path_common_words
        # with open(path_common_words) as f:
        #     self.common_words = json.load(f)
        # self.common_word_ratio_doc_level_min_cutoff = common_word_ratio_doc_level_min_cutoff

        self.cond_check_lang_id_doc_level = cond_check_lang_id_doc_level
        self.path_lang_id_model = path_lang_id_model
        if cond_check_lang_id_doc_level:
            self.lang_id_model = fasttext.load_model(path_lang_id_model)
        self.lang_id_doc_level_min_cutoff = lang_id_doc_level_min_cutoff

    def __call__(self, dict_md52text):
        dict_md52text_new = {}

        for ind, (md5, info) in tqdm(enumerate(dict_md52text.items()), total=len(dict_md52text)):
            full_text = info['Text']

            if self.cond_check_number_words_doc_level:
                if not FilteringFunctions.check_number_words(
                    text=full_text,
                    strip_characters=self.strip_characters,
                    number_words_min_cutoff=self.number_words_doc_level_min_cutoff,
                    number_words_max_cutoff=self.number_words_doc_level_max_cutoff,
                ):
                    continue

            if self.cond_check_character_repetition_ratio_doc_level:
                if not FilteringFunctions.check_character_repetition_ratio(
                    text=full_text,
                    character_repetition_length=self.character_repetition_length_doc_level,
                    character_repetition_max_cutoff=self.character_repetition_doc_level_max_cutoff,
                ):
                    continue

            if self.cond_check_word_repetition_ratio_doc_level:
                if not FilteringFunctions.check_word_repetition_ratio(
                    text=full_text,
                    strip_characters=self.strip_characters,
                    word_repetition_length=self.word_repetition_length_doc_level,
                    word_repetition_max_cutoff=self.word_repetition_doc_level_max_cutoff,
                ):
                    continue

            if self.cond_check_special_character_ratio_doc_level:
                if not FilteringFunctions.check_special_character_ratio(
                    text=full_text,
                    special_characters=self.strip_characters,
                    special_character_ratio_max_cutoff=self.special_character_ratio_doc_level_max_cutoff,
                ):
                    continue

            # if self.cond_check_stopword_ratio_doc_level:
            #     if not FilteringFunctions.check_stopword_ratio(
            #         text=full_text,
            #         strip_characters=self.strip_characters,
            #         stopwords=self.stopwords,
            #         stopword_ratio_min_cutoff=self.stopword_ratio_doc_level_min_cutoff,
            #     ):
            #         continue

            if self.cond_check_flagged_word_ratio_doc_level:
                if not FilteringFunctions.check_flagged_word_ratio(
                    text=full_text,
                    strip_characters=self.strip_characters,
                    flagged_words=self.flagged_words,
                    flagged_word_ratio_max_cutoff=self.flagged_word_ratio_doc_level_max_cutoff,
                ):
                    continue

            # if self.cond_check_punctuation_ratio_doc_level:
            #     if not FilteringFunctions.check_punctuation_ratio(
            #         text=full_text,
            #         punctuation=self.punctuation,
            #         punctuation_ratio_min_cutoff=self.punctuation_ratio_doc_level_min_cutoff,
            #     ):
            #         continue
            #
            # if self.cond_check_common_word_ratio_doc_level:
            #     if not FilteringFunctions.check_common_word_ratio(
            #         text=full_text,
            #         strip_characters=self.strip_characters,
            #         common_words=self.common_words,
            #         common_word_ratio_min_cutoff=self.common_word_ratio_doc_level_min_cutoff,
            #     ):
            #         continue

            if self.cond_check_lang_id_doc_level:
                if not FilteringFunctions.check_lang_id(
                    text=full_text,
                    lang_id_model=self.lang_id_model,
                    target_lang_id="en",
                    lang_id_min_cutoff=self.lang_id_doc_level_min_cutoff,
                ):
                    continue

            dict_md52text_new[md5] = dict_md52text[md5].copy()

        return dict_md52text_new

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.cond_check_number_words_doc_level,
                self.strip_characters,
                self.number_words_doc_level_min_cutoff,
                self.number_words_doc_level_max_cutoff,
                self.cond_check_character_repetition_ratio_doc_level,
                self.character_repetition_length_doc_level,
                self.character_repetition_doc_level_max_cutoff,
                self.cond_check_word_repetition_ratio_doc_level,
                self.word_repetition_length_doc_level,
                self.word_repetition_doc_level_max_cutoff,
                self.cond_check_special_character_ratio_doc_level,
                self.special_character_ratio_doc_level_max_cutoff,
                # self.cond_check_stopword_ratio_doc_level,
                # self.stopwords,
                # self.stopword_ratio_doc_level_min_cutoff,
                self.cond_check_flagged_word_ratio_doc_level,
                self.flagged_words,
                self.flagged_word_ratio_doc_level_max_cutoff,
                # self.cond_check_punctuation_ratio_doc_level,
                # self.punctuation,
                # self.punctuation_ratio_doc_level_min_cutoff,
                # self.cond_check_common_word_ratio_doc_level,
                # self.path_common_words,
                # self.common_word_ratio_doc_level_min_cutoff,
                self.cond_check_lang_id_doc_level,
                self.path_lang_id_model,
                self.lang_id_doc_level_min_cutoff,
            ),
        )
