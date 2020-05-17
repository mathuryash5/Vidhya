import logging
import os
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlockBlobService

from config.config import Config


class DatasetUtils(object):

    @staticmethod
    def make_clickable(address):
        """
        Make the url click-able.
        :param address: URL.
        :return: HTTP link to the URL.
        """
        return '<a href="{0}">{0}</a>'.format(address)

    @staticmethod
    def preview(text):
        """
        Show only a preview of the text dataset.
        :param text: input list.
        :return: first 30 elements of the list.
        """
        return text[:30] + '...'

    @staticmethod
    def format_metadata(df):
        """
        Formats the input df.
        :param df: input df.
        :return: Formatted df with make_clickable and preview.
        """
        simple_schema = ['cord_uid', 'source_x', 'title', 'abstract', 'authors', 'full_text_file', 'url']
        format_ = {'title': DatasetUtils.preview, 'abstract': DatasetUtils.preview, 'authors': DatasetUtils.preview,
                   'url': DatasetUtils.make_clickable}
        df[simple_schema].head().style.format(format_)
        return df

    @staticmethod
    def download_microsoft_cord_19_dataset(dataset_filename, dataset_filepath):
        """
        Downloads Microsoft's CORD-19 Dataset.
        :param dataset_filename: dataset filename.
        :param dataset_filepath: location to store the dataset.
        """
        logging.info("Downloading Microsoft CORD-19 dataset ...")
        azure_storage_account_name = Config.get_config("azure_storage_account_name")
        azure_storage_sas_token = Config.get_config("azure_storage_sas_token")

        # Create a blob service
        blob_service = BlockBlobService(account_name=azure_storage_account_name, sas_token=azure_storage_sas_token)
        azure_container_name = Config.get_config("azure_container_name")

        blob_service.get_blob_to_path(container_name=azure_container_name, blob_name=dataset_filename,
                                      file_path=dataset_filepath)
        logging.info("Downloading Microsoft CORD-19 dataset completed ...")

    @staticmethod
    def get_microsoft_cord_19_dataset():
        """
        Returns dataset in pandas df.
        :return: Microsoft CORD-19 Dataset df.
        """
        resources_dir = Config.get_config("resources_dir")
        dataset_dir = Config.get_config("dataset_dir")
        microsoft_cord_19_dataset_filepath = Config.get_config("microsoft_cord_19_dataset_filepath")
        Path(os.path.join(resources_dir, dataset_dir, microsoft_cord_19_dataset_filepath)).mkdir(parents=True,
                                                                                                 exist_ok=True)
        microsoft_cord_19_dataset_filename = Config.get_config("microsoft_cord_19_dataset_filename")
        dataset_filepath = os.path.join(resources_dir, dataset_dir, microsoft_cord_19_dataset_filepath,
                                        microsoft_cord_19_dataset_filename)
        if os.path.exists(dataset_filepath):
            logging.info("Metadata already exists!")
        else:
            DatasetUtils.download_microsoft_cord_19_dataset(microsoft_cord_19_dataset_filename, dataset_filepath)
        df = pd.read_csv(dataset_filepath)
        df = DatasetUtils.format_metadata(df)
        return df
