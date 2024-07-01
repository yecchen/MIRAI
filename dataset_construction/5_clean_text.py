import os
import sqlite3
from tqdm import tqdm
import json
import sys
import argparse
import logging
import json
import yaml

sys.path.append('../')

from obelics.processors import WebDocumentFilteringDocLevel, WebDocumentFilteringNodeLevel
from obelics.utils import (
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    PUNCTUATION,
    SPECIAL_CHARACTERS,
    STOPWORDS,
)

DATA_DIR = '../data/text_raw'
output_directory = '../data/text_tmp'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description="Filtering of WebDocs.")
    parser.add_argument(
        "--path_web_document_dataset",
        type=str,
        default=os.path.join(output_directory, 'dict_md52text.json'),
        help="Path of the dataset containing the web documents.",
    )
    parser.add_argument(
        "--path_save_web_document_dataset_paragraph_filtered",
        type=str,
        default=os.path.join(output_directory, 'dict_md52text_paragraph_filtered.json'),
        help="The path to save the filtered web document dataset.",
    )
    parser.add_argument(
        "--path_save_web_document_dataset_document_filtered",
        type=str,
        default=os.path.join(output_directory, 'dict_md52text_document_filtered.json'),
        help="The path to save the filtered web document dataset.",
    )
    parser.add_argument(
        "--path_config_filter_web_documents",
        type=str,
        default="../obelics/configs/config_filter_web_documents.yaml",
        help="The path of the config file containing the filtering parameters.",
    )
    parser.add_argument(
        "--path_lang_id_model",
        type=str,
        default="../obelics/models/lid.176.bin",
        help="The path of the lang id model (FastText).",
    )
    parser.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="../obelics/models/en.sp.model",
        help="The path of the SentencePiece model.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # load and merge the downloaded web documents
    dict_md52text = {} # {md5: {'SOURCEURL', 'Title', 'Text'}
    filenames = os.listdir(DATA_DIR)
    for filename in filenames:
        filepath = os.path.join(DATA_DIR, filename)
        print(filepath)
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        table_name = 'sites_data'
        cursor.execute(f'SELECT URLID, MD5, Title, Text FROM {table_name}')
        records = cursor.fetchall()
        for idx, record in tqdm(enumerate(records), total=len(records)):
            d = {}
            d['SOURCEURL'] = record[0]
            d['Title'] = record[2]
            d['Text'] = record[3]
            dict_md52text[record[1]] = d.copy()

    json.dump(dict_md52text, open(os.path.join(output_directory, 'dict_md52text.json'), 'w'))
    print('dict_md52text.json saved, length:', len(dict_md52text))
    logger.info("Finished loading the web document dataset")

    # start cleaning the text
    logger.info("Start cleaning the text")
    with open(args.path_config_filter_web_documents) as f:
        filtering_params = yaml.load(f, Loader=yaml.FullLoader)

    web_document_filtering_node_level = WebDocumentFilteringNodeLevel(
        cond_remove_non_printing_characters=filtering_params["cond_remove_non_printing_characters"],
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        cond_standardize_whitespace=filtering_params["cond_standardize_whitespace"],
        cond_check_number_words_node_level=filtering_params["cond_check_number_words_node_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_node_level_min_cutoff=filtering_params["number_words_node_level_min_cutoff"],
        number_words_node_level_max_cutoff=filtering_params["number_words_node_level_max_cutoff"],
        cond_check_character_repetition_ratio_node_level=filtering_params[
            "cond_check_character_repetition_ratio_node_level"
        ],
        character_repetition_length_node_level=filtering_params["character_repetition_length_node_level"],
        character_repetition_node_level_max_cutoff=filtering_params["character_repetition_node_level_max_cutoff"],
        cond_check_word_repetition_ratio_node_level=filtering_params["cond_check_word_repetition_ratio_node_level"],
        word_repetition_length_node_level=filtering_params["word_repetition_length_node_level"],
        word_repetition_node_level_max_cutoff=filtering_params["word_repetition_node_level_max_cutoff"],
        cond_check_special_character_ratio_node_level=filtering_params[
            "cond_check_special_character_ratio_node_level"
        ],
        special_character_ratio_node_level_max_cutoff=filtering_params[
            "special_character_ratio_node_level_max_cutoff"
        ],
        cond_check_flagged_word_ratio_node_level=filtering_params["cond_check_flagged_word_ratio_node_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_node_level_max_cutoff=filtering_params["flagged_word_ratio_node_level_max_cutoff"],
        cond_check_lang_id_node_level=filtering_params["cond_check_lang_id_node_level"],
        path_lang_id_model=args.path_lang_id_model,
        lang_id_node_level_min_cutoff=filtering_params["lang_id_node_level_min_cutoff"],
    )

    logger.info("Starting filtering the web document dataset at paragraph level")
    web_document_dataset_paragraph_filtered, n_p_total, n_p_remain = web_document_filtering_node_level(dict_md52text)
    logger.info("Finished filtering the web document dataset at paragraph level")

    logger.info("Starting saving the web document dataset paragraph-level filtered")
    json.dump(web_document_dataset_paragraph_filtered,
              open(args.path_save_web_document_dataset_paragraph_filtered, 'w'))
    logger.info("Finished saving the web document dataset paragraph-level filtered")

    logger.info(f"Number of documents in the raw web document dataset: {len(dict_md52text)}")
    logger.info(
        f"Number of documents in the paragraph-level filtered web document dataset: {len(web_document_dataset_paragraph_filtered)}")
    logger.info(f"Number of paragraphs in the raw web document dataset: {n_p_total}")
    logger.info(f"Number of paragraphs in the filtered web document dataset: {n_p_remain}")
    logger.info(f"{n_p_remain / n_p_total}")

    web_document_filtering_doc_level = WebDocumentFilteringDocLevel(
        cond_check_number_words_doc_level=filtering_params["cond_check_number_words_doc_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_doc_level_min_cutoff=filtering_params["number_words_doc_level_min_cutoff"],
        number_words_doc_level_max_cutoff=filtering_params["number_words_doc_level_max_cutoff"],
        cond_check_character_repetition_ratio_doc_level=filtering_params[
            "cond_check_character_repetition_ratio_doc_level"
        ],
        character_repetition_length_doc_level=filtering_params["character_repetition_length_doc_level"],
        character_repetition_doc_level_max_cutoff=filtering_params["character_repetition_doc_level_max_cutoff"],
        cond_check_word_repetition_ratio_doc_level=filtering_params["cond_check_word_repetition_ratio_doc_level"],
        word_repetition_length_doc_level=filtering_params["word_repetition_length_doc_level"],
        word_repetition_doc_level_max_cutoff=filtering_params["word_repetition_doc_level_max_cutoff"],
        cond_check_special_character_ratio_doc_level=filtering_params["cond_check_special_character_ratio_doc_level"],
        special_character_ratio_doc_level_max_cutoff=filtering_params["special_character_ratio_doc_level_max_cutoff"],
        cond_check_flagged_word_ratio_doc_level=filtering_params["cond_check_flagged_word_ratio_doc_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_doc_level_max_cutoff=filtering_params["flagged_word_ratio_doc_level_max_cutoff"],
        cond_check_lang_id_doc_level=filtering_params["cond_check_lang_id_doc_level"],
        path_lang_id_model=args.path_lang_id_model,
        lang_id_doc_level_min_cutoff=filtering_params["lang_id_doc_level_min_cutoff"],
    )

    logger.info("Starting filtering the web document dataset at document level")
    web_document_dataset_filtered = web_document_filtering_doc_level(web_document_dataset_paragraph_filtered)
    logger.info("Finished filtering the web document dataset at document level")

    logger.info("Starting saving the web document dataset document-level filtered")
    json.dump(web_document_dataset_filtered, open(args.path_save_web_document_dataset_document_filtered, 'w'))
    logger.info("Finished saving the web document dataset document-level filtered")

    logger.info(f"Number of documents in the raw web document dataset: {len(dict_md52text)}")
    logger.info(
        f"Number of documents in the paragraph-level filtered web document dataset: {len(web_document_dataset_paragraph_filtered)}")
    logger.info(
        f"Number of documents in the document-level filtered web document dataset: {len(web_document_dataset_filtered)}")

