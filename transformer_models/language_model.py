import logging
import os

import torch
from transformers import AutoTokenizer, AutoModel

from config.config import Config


class LanguageModel:

    model = None
    tokenizer = None
    device = None

    @staticmethod
    def load_model():
        logging.debug("Initializing transformer_models")
        pretrained_models_path = Config.get_config("pretrained_models_path")
        model_name = Config.get_config("model_name")
        model_path = os.path.join(pretrained_models_path, model_name)

        LanguageModel.tokenizer = AutoTokenizer.from_pretrained(model_path)
        LanguageModel.model = AutoModel.from_pretrained(model_path)
        LanguageModel.device = 'cpu'

        LanguageModel.model = LanguageModel.model.to(LanguageModel.device)
        LanguageModel.model.eval()

    @staticmethod
    def get_sentence_embeddings_from_sentence(sentence):
        """
        Generates sentence level embeddings for an input sentence.
        :param sentence: Input sentence.
        :return: Sentence level embeddings.
        """
        input_ids = LanguageModel.tokenizer.encode(sentence)
        input_ids = torch.LongTensor(input_ids)
        input_ids = input_ids.to(LanguageModel.device)

        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            out = LanguageModel.model(input_ids)

        hidden_states = out[2]
        sentence_embeddings = torch.mean(hidden_states[-1], dim=1).squeeze()
        return sentence_embeddings
