""" Generalized steps used in processing jobs

This object must be extended, and the `process` method overridden.

"""

import argparse
import platform
import time
import nltk
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T

from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession
from pyspark.broadcast import Broadcast
from validations import check_date, check_datetime
from urllib.parse import unquote
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords as sw

import contractions, text_helpers, toToken, synonyms, inflections
tweet_tokenizer = TweetTokenizer(preserve_case=False)


###############################################################################
#                               Helper Methods                                #
###############################################################################

def synonym_df_to_dict(df):
    """ Convert CSV to dictionary of word mappings for synonyms
    """
    # create synonym mappings as dictionary {'synonym1': 'term', 'synonym2': 'term'}
    d = {}
    for row in df.collect():
        head, *tail = row[0].split(",")
        for syn in tail:
            if syn:
                d[syn] = head
    return d


def pre_process_choices(text, br_args, br_stopwords, br_synonyms, br_lemmas, br_lang):
    """ Applies preprosseing filters to the text
    This methods goes through all the --preprocessing_choices
    that are passed in, and then filters down the tweet
    accordingly.
    :param text: tweet string
    :param args: dict of all the input parameters
    :returns: a list contianing tokens which have been pre-processed.
    """

    if text:

        # setup bools
        remove_urls = False
        remove_hashtags = False
        remove_emoji = False

        # retrieve topics from the broadcast variable
        if type(br_args) == Broadcast:
            args = br_args.value
        else:
            args = br_args

        # retrieve synonyms, lemmas and stopwords from
        # the broadcast variables
        syn_dict = br_synonyms.value
        stop_words_dict = br_stopwords.value
        lemmas_dict = br_lemmas.value
        lang = br_lang.value

        # lowercase
        # arabic does not have capital letters
        if ("lowercase" in args.preprocessing_choices or syn_dict) and lang != "ar":
            text = text.lower()

        # cleanup tweets by default
        text = text_helpers.tweet_text_cleanup(text)

        # expand contraction
        if ("contractions" in args.preprocessing_choices or syn_dict) and (lang == "en"):
            text = contractions.expand(text,
                                       drop_ownership=args.drop_contraction_ownership)

        # replace synonyms
        # TODO check support for arabic and spanish
        if syn_dict:
            text = synonyms.replace(text, syn_dict)

        # replace lemmas if broadcast variable has been created
        if args.lemmas_file:
            text = synonyms.replace(text, lemmas_dict)

        # remove punctuations
        # this method should work for arabic and spanish as well
        if "punctuation" in args.preprocessing_choices:
            text = text_helpers.remove_punctuation(text)

        # tokenize - is also needed for other options
        if "lemmatize" in args.preprocessing_choices or syn_dict or stop_words_dict:
            if lang != "en":
                text = toToken.multi_language_tokenizer(text, lang)
            else:
                text = tweet_tokenizer.tokenize(text)

        # further filter down the tokens
        if "hashtags" in args.preprocessing_choices:
            remove_hashtags = True

        if "emojis" in args.preprocessing_choices:
            remove_emoji = True

        if "urls" in args.preprocessing_choices:
            remove_urls = True

        if any([remove_urls, remove_hashtags, remove_emoji]) or (len(stop_words_dict) >= 1):
            text = toToken.filter_tokens(
                text,
                stopwords=br_stopwords,
                urls=remove_urls,
                remove_emojis=remove_emoji,
                filter_hashtags=remove_hashtags
            )

        # stemmed
        if "stemmed" in args.preprocessing_choices:
            if lang == "es":
                text = inflections.stem(text, 'es')
            elif lang == "ar":
                text = inflections.stem(text, 'ar')
            elif lang == "fr":
                text = inflections.stem(text, 'fr')
            else:
                text = inflections.stem(text)

        # if both Lemmatize and Stopwords
        if all(i in args.preprocessing_choices for i in ['lemmatize', 'stopwords']):
            text = inflections.lemmatize(text)

            text = toToken.filter_tokens(
                text,
                stopwords=br_stopwords,
                urls=remove_urls,
                remove_emojis=remove_emoji,
                filter_hashtags=remove_hashtags
            )

        # lemmatize
        elif "lemmatize" in args.preprocessing_choices:
            text = inflections.lemmatize(text)

        # TODO : temporary workaround to ensure preprocessing returns a string
        if isinstance(text, list):
            text = ' '.join(text)

        return text

    else:
        return ""


def generate_summary_file(spark, record_count, args, end_time):
    """ This method generates a summary file, by including the
        values of the input parameters and other specs
    """

    # setting up the first col to initialize the DF
    summary_df = spark.createDataFrame([("Value")], "string").toDF("Description")

    summary_df = summary_df.withColumn('total record count', F.lit(record_count))

    # get and add system details
    sys_details = str(platform.system()) + " " + str(platform.release())
    summary_df = summary_df.withColumn('System details', F.lit(sys_details))

    # add run time
    summary_df = summary_df.withColumn('Run time', F.lit(end_time))

    # add input
    summary_df = summary_df.withColumn('Input', F.lit(" ".join(str(x) for x in args.input)))

    # add output file name
    summary_df = summary_df.withColumn('Output file', F.lit(args.output))

    # add date ranges
    if args.start_date:
        summary_df = summary_df.withColumn('Start date', F.lit(args.start_date))

    if args.end_date:
        summary_df = summary_df.withColumn('End date', F.lit(args.end_date))

    # add lang param
    if args.lang:
        summary_df = summary_df.withColumn('Filtered lang', F.lit(args.lang))

    # add twitter params
    if args.dataset.startswith("twitter"):
        summary_df = summary_df.withColumn(
            'Tweet Types',
            F.lit(" ".join(str(x) for x in args.tweet_types))
        )

    # add TV params
    if args.tv_outlet:
        summary_df = summary_df.withColumn(
            'TV Channel',
            F.lit(" ".join(str(x) for x in args.tv_outlet))
        )

    # add Newspaper params
    if args.dataset.startswith("event_registry"):

        if args.news_region:
            summary_df = summary_df.withColumn(
                'Newspaper region',
                F.lit(" ".join(str(x) for x in args.news_region))
            )

        if args.news_division:
            summary_df = summary_df.withColumn(
                'Newspaper division',
                F.lit(" ".join(str(x) for x in args.news_division))
            )

        if args.news_states:
            summary_df = summary_df.withColumn(
                'Newspaper states',
                F.lit(" ".join(str(x) for x in args.news_states))
            )

        if args.news_concepts:
            summary_df = summary_df.withColumn(
                'Newspaper concepts',
                F.lit(" ".join(str(x) for x in args.news_concepts))
            )

    # setup summary filename
    summary_output = args.output.split(".csv")[0]
    summary_output += "_summary.csv"

    summary_df.toPandas().to_csv(summary_output, index=False)


def sc_path_exists(sc, path):
    """ Helper method to check if a path is readable by the Spark Context
    """
    try:
        sc.textFile(path).take(1)
        return True
    except Py4JJavaError:
        print("WARNING] - unable to read path: ", path)
        return False


###############################################################################
#                         Dataset Attribute Mappings                          #
###############################################################################

# Attributes on the dataframe must be normalized to a standard format,
# regardless of the input dataset. For example, Twitter Twint, and Twitter
# Premium API data must be converted to have the date formatted similarly.

def _normalize_twitter_text_attrs(df):
    """
    """

    # intersection of text columns (ie. text columns that are present on the
    # dataframe)
    text_attributes = ["full_text", "text", "tweet", "body"]
    present_text_attributes = list(set(df.columns) & set(text_attributes))

    # chain together `when` conditionals for present text attributes
    if present_text_attributes:

        text_when_conditional = None
        for attr in present_text_attributes:
            if text_when_conditional is None:
                text_when_conditional = F.when(F.col(attr).isNotNull(), F.col(attr))
            else:
                text_when_conditional = text_when_conditional.when(F.col(attr).isNotNull(), F.col(attr))

        # conditionally set `full_text` based on text attributes that are present on
        # the dataframe
        df = df.withColumn("full_text", text_when_conditional)

    return df


def _normalize_twitter_date_attrs(df):
    """
    """

    # normalize the date
    if "created_at" in df.columns:
        df = df.withColumn(
            "date",
            F.when(
                # Twint `created_at` date format
                F.col("created_at").rlike(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC"),
                F.to_date(F.substring(F.col("created_at"), 0, 19), 'yyyy-MM-dd HH:mm:ss')
            ).when(
                # Converted Twitter `created_at` date format
                F.col("created_at").rlike(r"\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \+0000 \d{4}"),
                F.to_date(F.substring(F.col("created_at"), 5, 30), "MMM dd HH:mm:ss Z yyyy")
            )
        )

        # add the timestamp
        df = df.withColumn(
            "timestamp",
            F.when(
                # Twint `created_at` date format
                F.col("created_at").rlike(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC"),
                F.from_utc_timestamp(F.col("created_at"), tz="Z")
            ).when(
                # Converted Twitter `created_at` date format
                F.col("created_at").rlike(r"\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \+0000 \d{4}"),
                F.to_timestamp(F.substring(F.col("created_at"), 5, 30), "MMM dd HH:mm:ss Z yyyy")
            )
        )

    # normalize decahose date
    if "postedTime" in df.columns and "date" in df.columns:
        df = df.withColumn(
            "date",
            F.when(
                F.col("postedTime").isNotNull(),
                F.to_date(F.from_utc_timestamp(F.col("postedTime"), tz="Z"))
            ).when(
                F.col("date").isNotNull(), F.col("date")
            )
        )

        # add the timestamp
        df = df.withColumn(
            "timestamp",
            F.when(
                F.col("postedTime").isNotNull(),
                F.to_date(F.from_utc_timestamp(F.col("postedTime"), tz="Z"))
            ).when(
                F.col("timestamp").isNotNull(), F.col("timestamp")
            )
        )

    elif "postedTime" in df.columns:
        df = df.withColumn(
            "date",
            F.to_date(F.from_utc_timestamp(F.col("postedTime"), tz="Z"))
        )

        # add the timestamp
        df = df.withColumn(
            "timestamp",
            F.from_utc_timestamp(F.col("postedTime"), tz="Z")
        )

    return df


def _normalize_twitter_id_attrs(df):
    """
    """

    # normalize id -> id_str
    if "id_str" in df.columns and "id" in df.columns:
        df = df.withColumn(
            "id_str",
            F.when(F.col("id_str").isNotNull(), F.col("id_str"))
            .when(F.col("id").isNotNull(), F.col("id"))

        )
    elif "id" in df.columns:
        df = df.withColumn("id_str", F.col("id"))

    return df


def _normalize_twitter_lang_attrs(df):
    """
    """

    # Note: Converted doesn't have lang attr
    #       premium has "lang": null most of the time

    # intersection of lang columns (ie. lang columns that are present in the
    # dataframe)
    lang_attributes = ["lang", "language", "twitter_lang"]
    present_lang_attributes = list(set(df.columns) & set(lang_attributes))

    # chain together `when` conditionals for present lang attributes
    if present_lang_attributes:

        lang_when_conditional = None
        for attr in present_lang_attributes:
            if lang_when_conditional is None:
                lang_when_conditional = F.when(F.col(attr).isNotNull(), F.col(attr))
            else:
                lang_when_conditional = lang_when_conditional.when(F.col(attr).isNotNull(), F.col(attr))

        # conditionall set `lang` based on language attributes that are present in
        # the dataframe
        df = df.withColumn("lang", lang_when_conditional)

    return df


def _normalize_twitter_handle_attrs(df):

    # intersection of handle columns (ie. handle columns that are present in the
    # dataframe)

    # ["user.screen_name", "username", "user.name", "actor.preferredUsername"]

    present_handle_attributes = list()

    # considering twitter datasets that have "user.screen_name", "user.name" cols
    if "user" in df.columns:
        handle_attributes = ["screen_name", "name"]
        # df.columns doesn't see the nested "user" colums, so we add in the nested cols by selecting in to it
        present_handle_attributes = list(set(df.columns + (df.select("user.*").columns)) & set(handle_attributes))

        # we have to add "user." to the col name, so that F.col() `when` conditional
        # can see the column.
        # for now - hack way of adding "user." to col names
        for z, attr in enumerate(present_handle_attributes):
            present_handle_attributes[z] = "user."+str(attr)

    # considering twitter datasets that have "actor.preferredUsername" col
    if "actor" in df.columns:
        handle_attributes = ["preferredUsername"]
        # df.columns doesn't see the nested "actor" col, so we add in the nested col by selecting in to it
        present_handle_actor_attributes = list(
            set(df.columns + (df.select("actor.*").columns)) & set(handle_attributes))

        # we have to add "actor." to the col name, so that F.col()
        # in the `when` conditional can see the column.
        # for now - hack way of adding "actor." to col names
        # append to the present_handle_attributes list
        if present_handle_actor_attributes:
            for z, attr in enumerate(present_handle_actor_attributes):
                present_handle_attributes.append("actor."+str(attr))

    # considering twitter datasets that have the "username" col
    if "username" in df.columns:
        present_handle_username_attributes = list(set(df.columns) & set(["username"]))

        # append to the present_handle_attributes list
        if present_handle_username_attributes:
            present_handle_attributes.append(present_handle_username_attributes[0])

    if present_handle_attributes:

        # chain together `when` conditionals for present lang attributes
        handle_when_conditional = None
        for attr in present_handle_attributes:
            if handle_when_conditional is None:
                handle_when_conditional = F.when(F.col(attr).isNotNull(), F.col(attr))
            else:
                handle_when_conditional = handle_when_conditional.when(F.col(attr).isNotNull(), F.col(attr))

        # conditionall set `handle` based on handle attributes in the dataframe
        df = df.withColumn("handle", handle_when_conditional)

    return df


def normalize_twitter(df):
    """ Normalizes all twitter dataset attributes
    :param df: dataframe
    :returns: dataframe with normalized twitter attributes
    """
    df = _normalize_twitter_text_attrs(df)
    df = _normalize_twitter_date_attrs(df)
    df = _normalize_twitter_id_attrs(df)
    df = _normalize_twitter_lang_attrs(df)
    df = _normalize_twitter_handle_attrs(df)
    return df


def normalize_radio(df):
    df = df.withColumnRenamed("value", "full_text")
    df = df.withColumn(
        "id_str",
        F.element_at(F.split(F.input_file_name(), "/"), -1)
    )
    df = df.withColumn(
        "date",
        F.to_date(
            F.regexp_extract(F.input_file_name(), r"(\d{1,4}-\d{1,2}-\d{1,2})", 0),
            'yyyy-MM-dd'
        )
    )
    return df


def normalize_cnn_survey_2020(df):
    df = df.withColumn(
        "id_str",
        F.col("id").cast(T.StringType())
    )
    df = df.withColumn(
        "date",
        F.to_date(
            F.regexp_extract(
                F.input_file_name(),
                r"(\d{1,2}-\d{1,2}-\d{1,2})",
                0
            ),
            'M-d-yy'
        )
    )
    return df


def normalize_gallup_survey_2016(df):
    df = df.withColumn(
        "id_str",
        F.col("case_id").cast(T.StringType())
    )
    df = df.withColumn(
        "date",
        F.to_date("RESPONDENT_DATE")
    )
    return df


def normalize_event_registry(df):
    df = df.withColumn("id_str", F.col("uri"))
    df = df.withColumn("full_text", F.col("body"))
    return df


def normalize_reddit_comments(df):
    df = df.withColumn("id_str", F.col("id"))
    df = df.withColumn("full_text", F.col("body"))
    df = df.withColumn("date", F.to_date(F.from_unixtime(F.col("created_utc"))))
    return df


def normalize_tv_eyes(df):
    df = df.withColumn("id_str", F.col("guid"))
    df = df.withColumn("full_text", F.col("body"))
    df = df.withColumn(
        "date", F.to_date(F.col("date"), 'yyyy-MM-dd HH:mm:ss')
    )
    return df


# Mapping of datasets, to the function for their normalization
ATTRIBUTE_NORMALIZATION_MAPPINGS = {
    "twitter": normalize_twitter,
    "radio": normalize_radio,
    "cnn_survey_2020": normalize_cnn_survey_2020,
    "gallup_survey_2016": normalize_gallup_survey_2016,
    "event_registry": normalize_event_registry,
    "reddit_comments": normalize_reddit_comments,
    "tv_eyes": normalize_tv_eyes,
}

###############################################################################
#                             Base Job Definition                             #
###############################################################################


class BaseJob(object):

    def __init__(self, name):

        # define common command-line arguments
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--input", required=True, nargs="+",
            help="1 or more input paths to be read with Apache Spark"
        )
        self.parser.add_argument(
            "--output", required=True,
            help="directory to write resulting files"
        )

        group_start = self.parser.add_mutually_exclusive_group()
        group_start.add_argument(
            "--start_date", type=check_date,
            help="inclusive start date for filtering record timestamp"
        )
        group_start.add_argument(
            "--start_datetime", type=check_datetime,
            help="inclusive start datetime for filtering record timestamp"
        )

        group_end = self.parser.add_mutually_exclusive_group()
        group_end.add_argument(
            "--end_date", type=check_date,
            help="inclusive end date for filtering record timestamp"
        )
        group_end.add_argument(
            "--end_datetime", type=check_datetime,
            help="inclusive end datetime for filtering record timestamp"
        )

        self.parser.add_argument(
            "--target_attr", default="full_text",
            help="attribute that is the target of text processing"
        )
        self.parser.add_argument(
            "--dataset", default="twitter",
            choices=[
                'cnn_survey_2020',
                'gallup_survey_2016',
                'event_registry',
                'reddit_comments',
                'twitter',
                'tv_eyes',
                'radio',
            ],
            help="input data type, for data specific logic"
        )
        self.parser.add_argument(
            "--lang",
            help="language to include in filtering of input data"
        )
        self.parser.add_argument(
            "--summarize",
            action='store_true'
        )
        self.parser.add_argument(
            "--write_uncoalesced",
            action='store_true',
            help="optional parameter to write results to partial files"

        )
        self.parser.add_argument(
            "--drop_duplicates",
            action='store_true'
        )
        self.parser.add_argument(
            "--generate_summary_file",
            action='store_true'
        )
        self.parser.add_argument(
            "--preprocessing_choices", nargs="*",
            choices=['stopwords', 'punctuation', 'lemmatize', 'stemmed',
                     'lowercase', 'contractions', 'hashtags', 'emojis', 'urls'],
            help="filter/remove out options",
            default=[]
        )
        self.parser.add_argument(
            "--stopwords_file",
            help="optionally provide stopwords file"
        )
        self.parser.add_argument(
            "--synonyms_file",
            help="optionally replace words with their given synonyms"
        )
        self.parser.add_argument(
            "--lemmas_file",
            help="optionally replace words with their given synonyms (lemmas)"
        )
        self.parser.add_argument(
            "--drop_contraction_ownership",
            action='store_true',
            help="optionally remove `'s` after contraction expansion"
        )
        self.parser.add_argument(
            "--preprocessed_output",
            help="optionally write pre-processed text to file"
        )
        self.parser.add_argument(
            "--multi_lang_preprocessing",
            default="en",
            choices=[
                "ar",
                "es",
                "fr",
            ],
            help="language to preprocess in - only 'ar', 'en & 'es' are supported"
        )

        #######################
        #  Twitter Arguments  #
        #######################

        self.parser.add_argument(
            '--tweet_types', nargs="+",
            choices=['tweet', 'retweet', 'quote'],
            default=['tweet', 'retweet', 'quote'],
            help="filter of tweet type"
        )

        ##############################
        #  Event Registry Arguments  #
        ##############################

        self.parser.add_argument(
            '--news_region', nargs="+",
            help="optional Event-Registry-specific region filter"
        )
        self.parser.add_argument(
            '--news_division', nargs="+",
            help="optional Event-Registry-specific division filter"
        )
        self.parser.add_argument(
            '--news_states', nargs="+",
            help="optional Event-Registry-specific states filter"
        )

        self.parser.add_argument(
            '--news_metadata',
            default="gs://mdi-1-236513-build-artifacts/resources/event-registry_source_metadata.csv",
            help="news article source meta-data to be joined with articles"
        )

        self.parser.add_argument(
            '--news_concepts', nargs="+",
            choices=[
                "Joe Biden",
                "Donald Trump",
                "Vaccine",
                "Pandemic",
                "Coronavirus",
            ],
            help="news article have associated concept words, which can be used in filtering"
        )

        #################################
        #  TV Eyes Tanscript Arguments  #
        #################################

        self.parser.add_argument(
            '--tv_outlet', nargs="+",
            choices=[
                "CNN",
                "Fox News",
                "MSNBC",
            ],
            help="optional TV Eyes television outlet"
        )

        # start recording time
        self.start_time = time.time()

        # setup spark session
        self.spark = SparkSession \
            .builder \
            .appName(name) \
            .getOrCreate()

        # initialize spark context from session
        self.sc = self.spark.sparkContext

    def _read(self):
        """ load dataset to `self.df`
        """

        # Throw an error if lemmaitization, or contraction expansion is
        # requested for a non-English, as this is not supported.
        if self.args.multi_lang_preprocessing != 'en':
            if self.args.preprocessing_choices == 'lemmatize' or self.args.preprocessing_choices == 'contractions':

                print("Note: Set the --preprocessing_choices manually without including 'lemmatize' 'contractions' ")

                self.parser.error(
                    "preprocessing options of 'lemmatize' or 'contractions' are only supported for an English corpus"
                )

        # support both multiple-arguments for the input file, or pipe-delimited
        # inputs. If pipe-delimited inputs are provided, then split them to a
        # list
        if len(self.args.input) == 1:
            if "|" in self.args.input[0]:
                self.args.input = self.args.input[0].split("|")

        # If there is percent-encodings in the input parameters, then perform
        # decoding
        self.args.input = [unquote(i) for i in self.args.input]

        # Check that the path exists before reading in the list of files. If
        # this is not done, then the entire read will fail, even if only one
        # path does not exist.
        self.args.input = [f for f in self.args.input if sc_path_exists(self.sc, f)]

        # If all files end in `.txt` or `.csv` read accordingly, otherwise
        # default to reading newline-delimited JSON
        if all([f.endswith(".txt") for f in self.args.input]):
            self.df = self.spark.read.text(self.args.input, wholetext=True)

        # Support Parquet format
        elif all([f.endswith(".parquet") for f in self.args.input]):
            self.df = self.spark.read.parquet(*self.args.input)

        # CSVs are notorious to have different formats, and we want to
        # keep the BaseJob as general as possible. So if your CSVs are not
        # properly concatinated/formatted, consider using or modifying one
        # or more of these parameters in to the read.csv() method.
        # multiLine=True, quote="\"", escape="\""
        elif all([f.endswith(".csv") for f in self.args.input]):
            self.df = self.spark.read.csv(self.args.input, header=True)

        else:
            self.df = self.spark \
                .read \
                .option("multiline", False) \
                .json(self.args.input)

        # Attribute mappings depending on the dataset
        if self.args.dataset in ATTRIBUTE_NORMALIZATION_MAPPINGS:
            self.df = ATTRIBUTE_NORMALIZATION_MAPPINGS[self.args.dataset](self.df)

        ######################################################################
        #                    Dataset-Specific Filtering                      #
        ######################################################################

        # Event Registry specific processing & filtering
        if self.args.dataset == 'event_registry':

            # resources/event-registry_source_metadata.csv
            # https://docs.google.com/spreadsheets/d/1oreMaErmBK6x0sO3JC2GoiB8ln4HWaaXqDTz2APjWJ8/edit#gid=1561911081
            df_meta = self.spark.read.csv(self.args.news_metadata, header=True)

            # Join Event Registry with news source metadata
            self.df = self.df.join(df_meta, on=(self.df["source.uri"] == df_meta["Source URL"]), how="left_outer")

            if self.args.news_region:

                # support pipe-delimited arguments
                if len(self.args.news_region) == 1:
                    if "|" in self.args.news_region[0]:
                        self.args.news_region = self.args.news_region[0].split("|")

                # filter empty strings
                self.args.news_region = [s for s in self.args.news_region if s]

                # ensure filter is not empty
                if self.args.news_region:
                    self.df = self.df.where(
                        F.col("Region (Census)").isin(self.args.news_region)
                    )

            if self.args.news_division:

                # support pipe-delimited arguments
                if len(self.args.news_division) == 1:
                    if "|" in self.args.news_division[0]:
                        self.args.news_division = self.args.news_division[0].split("|")

                # filter empty strings
                self.args.news_division = [s for s in self.args.news_division if s]

                # ensure filter is not empty
                if self.args.news_division:
                    self.df = self.df.where(
                        F.col("Division (Census)").isin(self.args.news_division)
                    )

            if self.args.news_states:

                # filter empty strings
                self.args.news_states = [s for s in self.args.news_states if s]

                # ensure filter is not empty
                if self.args.news_states:
                    self.df = self.df.where(
                        F.col("State").isin(self.args.news_states)
                    )

            if self.args.news_concepts:
                # Find where the concepts associated with the article, overlap with
                # the concepts that were provided as an argument - converting the
                # arguments to an array of literal strings
                self.df = self.df.where(
                    F.arrays_overlap(
                        "concepts.label.eng",
                        F.array(*[F.lit(a) for a in self.args.news_concepts])
                    )
                )

        if self.args.dataset == 'tv_eyes':
            # filter by news outlet
            if self.args.tv_outlet:
                self.df = self.df.where(
                    F.col("outlet").isin(self.args.tv_outlet)
                )

        if self.args.dataset == 'twitter':

            # filter tweet-types if argument is present
            if self.args.tweet_types:

                # exclude retweets -- tweets where there is no `retweet_status`
                if "retweet" not in self.args.tweet_types:
                    if "retweeted_status" in self.df.columns:
                        self.df = self.df.filter(F.col("retweeted_status").isNull())

                # exclude quotes -- tweets where `is_quote_status` is false
                if "quote" not in self.args.tweet_types:
                    if "is_quote_status" in self.df.columns:
                        self.df = self.df.filter(~F.col("is_quote_status"))

                # exclude tweets -- tweets where `retweeted_status` is not
                # null, or `is_quote_status` is true
                if "tweet" not in self.args.tweet_types:

                    # if both `retweeted_status` and `is_quote_status` exist,
                    # include tweets where either there is a retweet status, or
                    # it is a quote
                    if ("retweeted_status" in self.df.columns) and ("is_quote_status" in self.df.columns):
                        self.df = self.df.filter(
                            F.col("retweeted_status").isNotNull() |
                            F.col("is_quote_status")
                        )

                    # if `retweeted_status` does not exist on the dataframe, then
                    # there do not exists any retweets in the dataset, therefore we
                    # only want to include tweets with `is_quote_status`
                    elif "retweeted_status" not in self.df.columns:
                        self.df = self.df.filter(
                            F.col("is_quote_status")
                        )

            # use `full_text` when it is present, otherwise use `text`
            # self.df = normalize_text_attribute(self.df)

        #########################
        #  Attribute Filtering  #
        #########################

        global br_lang
        br_lang = self.sc.broadcast("")

        # Filter language if argument is present
        if self.args.lang:
            br_lang = self.sc.broadcast(self.args.lang)
            if "lang" in self.df.columns:
                self.df = self.df.filter(F.col("lang") == self.args.lang)

        # else set the br_lang to the --multi_lang_preprocessing param
        elif self.args.multi_lang_preprocessing:
            br_lang = self.sc.broadcast(self.args.multi_lang_preprocessing)

        # allow child classes to access lang
        self.preprocess_lang = br_lang.value

        # filter start date by timestamp if available, otherwise by date
        if self.args.start_date:
            self.df = self.df.where(
                F.col("date") >= F.to_date(F.lit(self.args.start_date))
            )

        if self.args.start_datetime:
            if "timestamp" in self.df.columns:
                self.df = self.df.where(
                    F.col("timestamp") >= F.lit(self.args.start_datetime)
                )

        # filter end date by timestamp if available, otherwise by date
        if self.args.end_date:
            self.df = self.df.where(
                F.col("date") <= F.to_date(F.lit(self.args.end_date))
            )

        if self.args.end_datetime:
            if "timestamp" in self.df.columns:
                self.df = self.df.where(
                    F.col("timestamp") <= F.lit(self.args.end_datetime)
                )

        # drop duplicates
        if(self.args.drop_duplicates):
            self.df = self.df.drop_duplicates(subset=["id_str"])

        # Count the number of records - only when a summary file is asked for
        if(self.args.generate_summary_file):
            self.record_count = self.df.count()
            print("*******************************************************")
            print("No. of Records: ", self.record_count)
            print("*******************************************************")

    def preprocess(self):

        # broadcast dictionary of args
        global br_args, br_synonyms, br_stopwords, br_lemmas
        br_args = self.sc.broadcast(self.args)

        ####################################
        #  Stopword Replacement (optional) #
        ####################################

        # Load custom stopwords file or laod corpus from NLTK
        stopwords_list = []
        if self.args.stopwords_file:
            stopwords_df = pd.read_csv(self.args.stopwords_file, sep=" ")
            stopwords_list = [r[0] for r in stopwords_df.values.tolist()]
            # include `rt` and `amp` in stopwords list
            stopwords_list += ["rt", "amp"]
            print("Loaded stopwords file with # entries:", len(stopwords_list))

        elif "stopwords" in self.args.preprocessing_choices:
            # download the stopwords from NLTK
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords")

            if self.preprocess_lang == "ar":
                stopwords_list = sw.words('arabic')
            elif self.preprocess_lang == "es":
                stopwords_list = sw.words('spanish')
            elif self.preprocess_lang == "fr":
                stopwords_list = sw.words('french')
            else:
                stopwords_list = sw.words('english')
            # include `rt` and `amp` in stopwords list
            stopwords_list += ["rt", "amp"]

        br_stopwords = self.sc.broadcast(stopwords_list)

        ####################################
        #  Synonym Replacement (optional)  #
        ####################################

        #  load the custom file of synonyms
        syn_dict = []
        if self.args.synonyms_file:
            # read synonym file to dataframe
            df_synonyms = self.spark.read.text(self.args.synonyms_file)
            syn_dict = synonym_df_to_dict(df_synonyms)
            print("Loaded synonyms file with # entries:", len(syn_dict))

        # broadcast variable to be leveraged in the `pre_process_choices` function
        br_synonyms = self.sc.broadcast(syn_dict)

        ####################################
        #     Lemmatize (optional)         #
        ####################################

        # load the custom lemmas_file
        lemmas_dict = []
        if self.args.lemmas_file:
            # read lemas file to dataframe
            df_lemmas = self.spark.read.text(self.args.lemmas_file)
            lemmas_dict = synonym_df_to_dict(df_lemmas)
            print("Loaded lemmas file with # entries:", len(lemmas_dict))

        # broadcast variable to be leveraged in the `udf_tokenize` function
        br_lemmas = self.sc.broadcast(lemmas_dict)

        #########################
        #     Preprocessing     #
        #########################

        if any([lemmas_dict, syn_dict, stopwords_list]) or (len(self.args.preprocessing_choices) >= 1):

            # preprocess the 'full_text' column
            udf_pre_processing = F.udf(
                lambda r: pre_process_choices(r, br_args, br_stopwords, br_synonyms, br_lemmas, br_lang),
                T.StringType()
            )

            self.df = self.df.withColumn(
                "preprocessed_text",
                udf_pre_processing(F.col(self.args.target_attr))
            )

            # child jobs should target `preprocessed_text` if pre-processing
            # has taken place
            self.args.target_attr = "preprocessed_text"

        ####################################
        #  Preprocessed Output  (optional) #
        ####################################

        # write pre-processed text and tokens to output file
        if self.args.preprocessed_output:
            self.df \
                .withColumn(self.args.target_attr, F.trim(F.col(self.args.target_attr))) \
                .withColumn("preprocessed_text", F.trim(F.array_join("preprocessed_text", " "))) \
                .select("id_str", self.args.target_attr, "preprocessed_text") \
                .toPandas() \
                .to_json(self.args.preprocessed_output, orient="records", lines=True)

    def process(self):
        # operate on `self.df`
        raise NotImplementedError

    def _write(self):

        # if file does not have `.csv` extension, then append it
        if not self.args.output.endswith(".csv"):
            self.args.output += ".csv"

        # WARNING: this will collect the dataframe to the driver, and will likely
        # exceed heap memory limits unless aggregates have been computed!
        if self.df:
            print(f"writing output: {self.args.output}")
            if self.args.write_uncoalesced:
                self.df.write.json(self.args.output)
            else:
                # otherwise convert to pandas dataframe for single file output
                self.df \
                    .toPandas() \
                    .to_csv(self.args.output, index=False)

        # end recording time
        self.end_time = time.time() - self.start_time

        # generate summary file
        if (self.args.generate_summary_file):
            generate_summary_file(self.spark, self.record_count, self.args, self.end_time)

    def run(self):
        self.args = self.parser.parse_args()
        self._read()
        self.preprocess()
        self.process()
        self._write()