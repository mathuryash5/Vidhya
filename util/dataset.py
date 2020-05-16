from pathlib import Path

from azure.storage.blob import BlockBlobService
import os
import logging
import pandas as pd


from config.config import Config


class DatasetUtils(object):

    @staticmethod
    def make_clickable(address):
        """Make the url clickable"""
        return '<a href="{0}">{0}</a>'.format(address)

    @staticmethod
    def preview(text):
        """Show only a preview of the text data."""
        return text[:30] + '...'

    @staticmethod
    def format_metadata(df):
        simple_schema = ['cord_uid', 'source_x', 'title', 'abstract', 'authors', 'full_text_file', 'url']
        format_ = {'title': DatasetUtils.preview, 'abstract': DatasetUtils.preview, 'authors': DatasetUtils.preview, 'url': DatasetUtils.make_clickable}
        df[simple_schema].head().style.format(format_)
        return df

    @staticmethod
    def download_microsoft_metadata(metadata_filename, metadata_location):
        logging.info("Downloading metadata ...")
        azure_storage_account_name = Config.get_config("azure_storage_account_name")
        azure_storage_sas_token = Config.get_config("azure_storage_sas_token")

        # create a blob service
        blob_service = BlockBlobService(account_name=azure_storage_account_name, sas_token=azure_storage_sas_token)
        azure_container_name = Config.get_config("azure_container_name")

        blob_service.get_blob_to_path(container_name=azure_container_name, blob_name=metadata_filename,
                                      file_path=metadata_location)
        logging.info("Downloading metadata completed ...")

    @staticmethod
    def get_microsoft_metadata():
        data_filepath = Config.get_config("data_dir")
        metadata_filepath = Config.get_config("microsoft_data")
        Path(os.path.join(data_filepath, metadata_filepath)).mkdir(parents=True, exist_ok=True)
        metadata_filename = Config.get_config("microsoft_metadata")
        metadata_location = os.path.join(data_filepath, metadata_filepath, metadata_filename)
        if os.path.exists(metadata_location):
            logging.info("Metadata already exists!")
        else:
            DatasetUtils.download_microsoft_metadata(metadata_filename, metadata_location)
            logging.info("Metadata downloaded successfully")
        df = pd.read_csv(metadata_location)
        df = DatasetUtils.format_metadata(df)

        return df
