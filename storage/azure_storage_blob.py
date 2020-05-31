import os
import sys
import uuid
import logging
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, PublicAccess
from config.config import Config
from pathlib import Path
from util.dataset import DatasetUtils


class MicrosoftAzureBlobStorageAPI:

    @staticmethod
    def setup_storage():
        DatasetUtils.get_microsoft_cord_19_dataset()
        MicrosoftAzureBlobStorageAPI.download_embeddings()
        MicrosoftAzureBlobStorageAPI.download_pretrained_models()
        MicrosoftAzureBlobStorageAPI.download_paper_sentence_text()

    @staticmethod
    def download_embeddings():
        logging.debug("Downloading embeddings started ...")

        conn_str = "DefaultEndpointsProtocol=https;AccountName=vidhyablobstorage;AccountKey=xqpTeFPmS1LBe6l8wT+QRuillPaieOVWqrjnANcQsWs7+DJz+ZwYWnwpmJH30k+KA139lsX3AdNViwr+U8tVsw==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

        container_name = "embeddings"
        blob_names = ["title_embeddings.pkl", "abstract_embeddings.pkl", "sentence_embeddings.pkl"]

        resources_dir = Config.get_config("resources_dir_key")
        generated_embeddings_dir = Config.get_config("embeddings_path_key")
        generated_embeddings_path = os.path.join(resources_dir, generated_embeddings_dir)

        Path(generated_embeddings_path).mkdir(parents=True, exist_ok=True)

        for blob_name in blob_names:
            download_path = os.path.join(generated_embeddings_path, blob_name)
            if os.path.exists(download_path):
                logging.debug(blob_name + " already exists!")
                continue
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(download_path, "wb") as blob:
                blob.writelines([blob_client.download_blob().readall()])
        
        logging.debug("Downloading embeddings completed...")
    
    @staticmethod
    def download_pretrained_models():
        logging.debug("Downloading pretrained models started ...")

        conn_str = "DefaultEndpointsProtocol=https;AccountName=vidhyablobstorage;AccountKey=xqpTeFPmS1LBe6l8wT+QRuillPaieOVWqrjnANcQsWs7+DJz+ZwYWnwpmJH30k+KA139lsX3AdNViwr+U8tVsw==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

        container_name = "pretrained-model"
        blob_names = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]

        pretrained_model_dir = Config.get_config("pretrained_models_path_key")
        model_name = Config.get_config("model_name_key")
        pretrained_model_path = os.path.join(pretrained_model_dir, model_name)

        Path(pretrained_model_path).mkdir(parents=True, exist_ok=True)

        for blob_name in blob_names:
            download_path = os.path.join(pretrained_model_path, blob_name)
            if os.path.exists(download_path):
                logging.debug(blob_name + " already exists!")
                continue
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(download_path, "wb") as blob:
                blob.writelines([blob_client.download_blob().readall()])
        
        logging.debug("Downloading pretrained models completed...")
    
    # CHECK
    @staticmethod
    def download_paper_sentence_text():
        logging.debug("Downloading paper sentence texts started ...")

        conn_str = "DefaultEndpointsProtocol=https;AccountName=vidhyablobstorage;AccountKey=xqpTeFPmS1LBe6l8wT+QRuillPaieOVWqrjnANcQsWs7+DJz+ZwYWnwpmJH30k+KA139lsX3AdNViwr+U8tVsw==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)
        container_name = "paper-sentence-text"
        container_client = blob_service_client.get_container_client(container_name)
        
        generator = container_client.list_blobs()

        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("embeddings_path_key")
        paper_sentence_text_dir = Config.get_config("paper_sentence_text_dir_key")
        paper_sentence_path = os.path.join(resources_dir, dataset_dir, paper_sentence_text_dir)

        if not os.path.exists(paper_sentence_path):

            Path(paper_sentence_path).mkdir(parents=True, exist_ok=True)
            paper_sentence_path = os.path.join(resources_dir, dataset_dir)
            for blob_name in generator:
                # logging.debug("Blob: {}".format(blob_name))
                download_path = os.path.join(paper_sentence_path, blob_name.name)
                if os.path.exists(download_path):
                    logging.debug(blob_name.name + " already exists!")
                    continue
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name.name)
                with open(download_path, "wb") as blob:
                    blob.writelines([blob_client.download_blob().readall()])
        
        logging.debug("Downloading paper sentence texts completed...")