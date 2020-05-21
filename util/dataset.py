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
        df.fillna("", inplace=True)
        return df

    @staticmethod
    def get_papers_text(df):
        azure_storage_account_name = Config.get_config("azure_storage_account_name")
        azure_storage_sas_token = Config.get_config("azure_storage_sas_token")

        # Create a blob service
        blob_service = BlockBlobService(account_name=azure_storage_account_name, sas_token=azure_storage_sas_token)
        azure_container_name = Config.get_config("azure_container_name")

        for index, row in df.iterrows():
            if row['has_pdf_parse']:
                text = ""
                try:
                    blob_name = '{0}/pdf_json/{1}.json'.format(row['full_text_file'],
                                                           row['sha'])  # note the repetition in the path
                    logging.debug("Full text blob for this entry: {}".format(blob_name))
                    blob_as_json_string = blob_service.get_blob_to_text(container_name=azure_container_name,
                                                                        blob_name=blob_name)
                    data = json.loads(blob_as_json_string.content)
                    text_and_metadata = data['body_text']
                    for paragraph in text_and_metadata:
                        text += paragraph['text']
                except Exception as e:
                    pass

                filename = "resources/dataset/microsoft/paper_text/" + row['cord_uid'] + ".txt"
                with open(filename, "w") as file:
                    file.write(text)
            elif row['has_pmc_xml_parse']:
                text = ""
                # construct path to blob containing full text
                try:
                    blob_name = '{0}/pmc_json/{1}.xml.json'.format(row['full_text_file'], row[
                        'pmcid'])  # note the repetition in the path
                    logging.debug("Full text blob for this entry: {}".format(blob_name))
                    blob_as_json_string = blob_service.get_blob_to_text(container_name=azure_container_name, blob_name=blob_name)
                    data = json.loads(blob_as_json_string.content)
                    text_and_metadata = data['body_text']
                    for paragraph in text_and_metadata:
                        text += paragraph['text']
                except Exception as e:
                    pass
                filename = "resources/dataset/microsoft/paper_text/" + row['cord_uid'] + ".txt"
                with open(filename, "w") as file:
                    file.write(text)
            else:
                text = ""
                filename = "resources/dataset/microsoft/paper_text/" + row['cord_uid'] + ".txt"
                with open(filename, "w") as file:
                    file.write(text)



