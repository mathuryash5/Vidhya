import logging
import os
import pickle

import torch
from transformers import AutoTokenizer, AutoModel

from config.config import Config
from util.dataset import DatasetUtils


class BioBERT:

    @staticmethod
    def get_sentence_embeddings(sentence):
        pretrained_models_path = Config.get_config("pretrained_models_path")
        model_name = Config.get_config("model_name")
        model_path = os.path.join(pretrained_models_path, model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        input_ids = tokenizer.encode(sentence)
        input_ids = torch.LongTensor(input_ids)

        model = AutoModel.from_pretrained(model_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        input_ids = input_ids.to(device)

        model.eval()

        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids)

        hidden_states = out[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        logging.debug("Sentence embedding generated for sentence = {} \n Tensor Size = {}".
                      format(sentence, sentence_embedding.size()))
        return sentence_embedding

    @staticmethod
    def _get_multiple_sentence_embeddings(uid_sentence_mapping):
        uid_sentence_embedding_mapping = dict()
        for uid, sentence in uid_sentence_mapping.items():
            uid_sentence_embedding_mapping[uid] = BioBERT.get_sentence_embeddings(sentence)
        return uid_sentence_embedding_mapping

    @staticmethod
    def generate_and_store_embeddings():
        # these functions should be in some other file
        # load the model once and not on every function call
        embeddings_path = Config.get_config("embeddings_path")
        title_embeddings_filename = Config.get_config("title_embeddings_filename")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename")
        title_embeddings_path = os.path.join(embeddings_path, title_embeddings_filename)
        abstract_embeddings_path = os.path.join(embeddings_path, abstract_embeddings_filename)
        metadata_df = DatasetUtils.get_microsoft_metadata()

        if os.path.exists(title_embeddings_path):
            logging.info("Title Embeddings have already been generated")
        else:
            uid_title_mapping = dict()
            for index, row in metadata_df.iterrows():
                uid_title_mapping[row["cord_uid"]] = row["title"]
                break
            logging.info("Generating sentence embeddings for titles...")
            uid_title_embedding_mapping = BioBERT._get_multiple_sentence_embeddings(uid_title_mapping)
            with open(title_embeddings_path, 'wb') as handle:
                pickle.dump(uid_title_embedding_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if os.path.exists(abstract_embeddings_path):
        #     logging.info("Abstract Embeddings have already been generated")
        # else:
        #     uid_abstract_mapping = dict()
        #     for index, row in metadata_df.iterrows():
        #         uid_abstract_mapping[row["cord_uid"]] = row["abstract"]
        #     logging.info("Generating sentence embeddings for abstracts...")
        #     uid_abstract_embedding_mapping = BioBERT._get_multiple_sentence_embeddings(uid_abstract_mapping)

        # dump these dictionaries into pickle files
