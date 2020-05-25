import json
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
        azure_storage_account_name = Config.get_config("azure_storage_account_name_key")
        azure_storage_sas_token = Config.get_config("azure_storage_sas_token_key")

        # Create a blob service
        blob_service = BlockBlobService(account_name=azure_storage_account_name, sas_token=azure_storage_sas_token)
        azure_container_name = Config.get_config("azure_container_name_key")

        blob_service.get_blob_to_path(container_name=azure_container_name, blob_name=dataset_filename,
                                      file_path=dataset_filepath)
        logging.info("Downloading Microsoft CORD-19 dataset completed ...")

    @staticmethod
    def get_microsoft_cord_19_dataset():
        """
        Returns dataset in pandas df.
        :return: Microsoft CORD-19 Dataset df.
        """
        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("dataset_dir_key")
        microsoft_cord_19_dataset_filepath = Config.get_config("microsoft_cord_19_dataset_filepath_key")
        Path(os.path.join(resources_dir, dataset_dir, microsoft_cord_19_dataset_filepath)).mkdir(parents=True,
                                                                                                 exist_ok=True)
        microsoft_cord_19_dataset_filename = Config.get_config("microsoft_cord_19_dataset_filename_key")
        dataset_filepath = os.path.join(resources_dir, dataset_dir, microsoft_cord_19_dataset_filepath,
                                        microsoft_cord_19_dataset_filename)
        if os.path.exists(dataset_filepath):
            logging.info("Microsoft CORD-19 dataset already exists!")
        else:
            DatasetUtils.download_microsoft_cord_19_dataset(microsoft_cord_19_dataset_filename, dataset_filepath)
        df = pd.read_csv(dataset_filepath)
        df = DatasetUtils.format_metadata(df)
        df.fillna("", inplace=True)
        return df

    @staticmethod
    def download_microsoft_cord_19_paper_text(df):
        """
        Downloads Microsoft's CORD-19 Dataset.
        :param df: Microsoft's CORD-19 Dataset df.
        """
        azure_storage_account_name = Config.get_config("azure_storage_account_name_key")
        azure_storage_sas_token = Config.get_config("azure_storage_sas_token_key")

        # Create a blob service
        blob_service = BlockBlobService(account_name=azure_storage_account_name, sas_token=azure_storage_sas_token)
        azure_container_name = Config.get_config("azure_container_name_key")

        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("dataset_dir_key")
        microsoft_dir = Config.get_config("microsoft_dir_key")
        paper_text_dir = Config.get_config("paper_text_dir_key")
        dir = os.path.join(resources_dir, dataset_dir, microsoft_dir, paper_text_dir)
        os.mkdir(dir)

        logging.info("Downloading Microsoft CORD-19 dataset paper texts ...")

        for index, row in df.iterrows():
            if row[Config.get_config("has_pdf_parse_key")]:
                text = ""
                try:
                    blob_name = '{0}/pdf_json/{1}.json'.format(row[Config.get_config("full_text_file_key")],
                                                           row[Config.get_config("sha_key")])  # note the repetition in the path
                    logging.debug("Full text blob for this entry: {}".format(blob_name))
                    blob_as_json_string = blob_service.get_blob_to_text(container_name=azure_container_name,
                                                                        blob_name=blob_name)
                    data = json.loads(blob_as_json_string.content)
                    text_and_metadata = data[Config.get_config("body_text_key")]
                    for paragraph in text_and_metadata:
                        text += paragraph[Config.get_config("text_key")]
                except Exception as e:
                    pass
                filename = os.path.join(dir, row[Config.get_config("cord_uid_key")] + ".txt")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)
            elif row[Config.get_config("has_pmc_xml_parse_key")]:
                text = ""
                # construct path to blob containing full text
                try:
                    blob_name = '{0}/pmc_json/{1}.xml.json'.format(row[Config.get_config("full_text_file_key")], row[
                        Config.get_config("pmcid_key")])  # note the repetition in the path
                    logging.debug("Full text blob for this entry: {}".format(blob_name))
                    blob_as_json_string = blob_service.get_blob_to_text(container_name=azure_container_name, blob_name=blob_name)
                    data = json.loads(blob_as_json_string.content)
                    text_and_metadata = data[Config.get_config("body_text_key")]
                    for paragraph in text_and_metadata:
                        text += paragraph[Config.get_config("text_key")]
                except Exception as e:
                    pass
                filename = os.path.join(dir, row[Config.get_config("cord_uid_key")] + ".txt")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)
            else:
                text = ""
                filename = os.path.join(dir, row[Config.get_config("cord_uid_key")] + ".txt")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)

        logging.info("Downloading Microsoft CORD-19 dataset paper texts completed...")


    @staticmethod
    def get_microsoft_cord_19_paper_text(df):
        """
        Checks if Microsoft CORD 19 paper text is downloaded
        or not.
        :param df: Microsoft's CORD-19 Dataset df.
        """
        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("dataset_dir_key")
        microsoft_dir = Config.get_config("microsoft_dir_key")
        paper_text_dir = Config.get_config("paper_text_dir_key")
        dir = os.path.join(resources_dir, dataset_dir, microsoft_dir, paper_text_dir)
        if os.path.exists(dir):
            logging.info("Microsoft CORD-19 dataset paper texts already exists!")
        else:
            df = DatasetUtils.get_microsoft_cord_19_dataset()
            DatasetUtils.download_microsoft_cord_19_paper_text(df)
