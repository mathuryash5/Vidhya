import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelWithLMHead, AutoModel, PretrainedConfig

from config.config import Config
import os
import logging


class BioBERT:

    @staticmethod
    def get_sentence_embeddings(sentence):
        pretrained_models_path = Config.get_config("pretrained_models_path")
        model_name = Config.get_config("model_name")
        model_path = os.path.join(pretrained_models_path, model_name)

        tokenizer = AutoTokenizer.from_pretrained("./resources/pretrained_models/biobert-nli/")

        input_ids = tokenizer.encode(sentence)
        input_ids = torch.LongTensor(input_ids)

        model = AutoModel.from_pretrained("./resources/pretrained_models/biobert-nli/")


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        input_ids = input_ids.to(device)

        model.eval()

        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids)

        hidden_states = out[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        logging.debug("Sentence embedding generated for sentence = {} shape = {}".format(sentence,-
                                                                                         sentence_embedding.size()))
        return sentence_embedding
