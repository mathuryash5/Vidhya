from html import unescape

from config.config import Config
from util.dataset import DatasetUtils


class FormatUtils:

    # Common formatter

    @staticmethod
    def capitalize_df_columns(df):
        """
        Capitalize df columns.
        :param df: Input df.
        :return: Formatted df.
        """
        df.columns = [str(column).lower().capitalize() for column in df.columns]
        return df

    # Paper recommender system specific formatter

    @staticmethod
    def paper_recommender_system_format_abstract_length(df):
        """
        Attaches read more/read less button to abstract
        if the length of abstract > 60.
        :param df: Input df.
        :return: Formatted df.
        """
        row_count = 0
        for index, row in df.iterrows():
            abstract = row[Config.get_config("paper_recommender_system_df_abstract_key")]
            abstract_len = len(abstract)
            if abstract_len > 100:
                abstract_split_1 = r'<p>' + abstract[
                                            :100] + r'<span id="dots{}">...</span><span id="more{}" style="display:none;">'.format(
                    str(row_count), str(row_count))
                abstract_split_2 = abstract[
                                   100:abstract_len] + r'</span></p><a onclick="myFunction({})" id="myBtn{}" style="color:blue">Read more</a>'.format(
                    str(row_count), str(row_count))
                formatted_abstract = abstract_split_1 + abstract_split_2
                df.at[index, Config.get_config("paper_recommender_system_df_abstract_key")] = formatted_abstract
                row_count += 1
        df[Config.get_config("paper_recommender_system_df_abstract_key")] = df[Config.get_config("paper_recommender_system_df_abstract_key")].apply(unescape)
        return df

    @staticmethod
    def paper_recommender_system_format_url_clickable(df):
        for index, row in df.iterrows():
            url = row[Config.get_config("paper_recommender_system_df_url_key")]
            url = '<div class="span2"><a class="btn btn-primary" href="{}" role="button" id="url">Link to paper</a></div>'.format(url)
            df.at[index, Config.get_config("paper_recommender_system_df_url_key")] = url
        df[Config.get_config("paper_recommender_system_df_url_key")] = df[Config.get_config("paper_recommender_system_df_url_key")].apply(unescape)
        return df

    # Paper recommender system formatter

    @staticmethod
    def paper_recommender_system_formatter(df):
        """
        Formats df for paper recommender system.
        :param df: Input df.
        :return: Formatted df.
        """
        df = FormatUtils.capitalize_df_columns(df)
        df = FormatUtils.paper_recommender_system_format_abstract_length(df)
        df = FormatUtils.paper_recommender_system_format_url_clickable(df)
        return df