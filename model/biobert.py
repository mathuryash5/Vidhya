import torch
from transformers import BertTokenizer, BertModel

from config.config import Config
import os
import logging


class BioBERT:

    @staticmethod
    def get_sentence_embeddings(sentence):
        pretrained_models_path = Config.get_config("pretrained_models_path")
        model_name = Config.get_config("model_name")
        model_path = os.path.join(pretrained_models_path, model_name)

        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertModel.from_pretrained(model_name, output_hidden_states=True)

        input_ids = torch.tensor([tokenizer.encode(sentence)])

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model = model.to(device)
        # input_ids = input_ids.to(device)

        model.eval()

        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids)

        hidden_states = out[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        logging.debug("Sentence embedding generated for sentence = {} shape = {}".format(sentence,
                                                                                         sentence_embedding.shape))
        return sentence_embedding
