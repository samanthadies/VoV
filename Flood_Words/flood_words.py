""" Compute the co-occurring words in a given topic

Topics can be provided as either a JSON object on the command-line, or a JSON
file. The expected format is as an object that representes a topic word,
followed by a list of words within that topic:

    [
        {
            "topic": "economy",
            "words": [
                "market",
                "wall street",
                "funding"
            ]
        },
        {
            "topic": "emotion",
            "words": [
                "against",
                "positve",
                "negative"
            ]
        }
    ]

Results are output as a CSV:
date,topic, word, count
5/28/20	police	tonight	14
5/28/20	police	better	31
5/28/20	police	last	21
5/28/20	police	also	54
5/28/20	police	world	53
5/28/20	police	reminder	9
5/28/20	police	info	10

"""

import json
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd

from struct import Struct
from pyspark.broadcast import Broadcast
import toToken
from base_job import BaseJob
from job_helpers import construct_output_filename

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

# Schema for topic element in output
TOPIC_SCHEMA = T.ArrayType(
    T.StructType([
        T.StructField("topic", T.StringType(), True),
        T.StructField("word", T.ArrayType(T.StringType(), True), True),
        T.StructField("occurrence", T.IntegerType(), True),
    ])
)


@F.udf(returnType=T.ArrayType(T.StringType()))
def udf_tokenize(text, lang=""):
    """ Composition of text tokenization functions:
    :param text: text to process
    :lang: language to tokenize in
    """

    if text:
        if lang != "en":
            tokens = toToken.multi_language_tokenizer(text, lang)
        else:
            tokens = toToken.tokenize(text)
        return tokens

    return []


def topic_co_occurrence(tokens):
    """ Determine the frequency of theme words in a set of tokens
    :param tokens:
    :returns: a list [{'word': ['word(s)'], 'topic': 'theme', 'occurrence': 1}]
    """

    # retrieve topics from the broadcast variable
    if type(br_topics) == Broadcast:
        topics = br_topics.value
    else:
        topics = br_topics

    # dictionary to keep track of word occurrences
    freqs = {}

    # when a tweet contains one word, it can be represented as a
    # string, and because there is only one word, there will be no
    # co-occurrences.
    if isinstance(tokens, str):
        return []

    # iterate over tokens, and topics and look for word occurrences
    for topic in topics:

        # check for topic word occurrences in tokens
        for word in topic['words']:
            if word in tokens:

                # add all tokens that are not the matching word, these are
                # co-occurrences
                tokens = [token for token in tokens if token != word]

                if topic['topic'] in freqs:
                    freqs[topic['topic']]['word'] += tokens
                else:
                    freqs[topic['topic']] = {"topic": topic}
    result = []
    for t in freqs:
        freqs[t]['topic'] = t
        freqs[t]['occurrence'] = 1
        result.append(freqs[t])

    return result


def flood_word_contenders(df):
    """ Determine the words which occur with a high frequency in a high percentage of topics
    :param tokens:
    :returns: a list ['word']
    """

    if type( df.select("count").head() ) is not type( None ):

        max_value = df.select("count").head()[0]

        flood_word_freq_cutoff = .60 * max_value
        int_freq_cutoff = int(flood_word_freq_cutoff)
        df = df.filter(F.col("count") > int_freq_cutoff)

    # retrieve topics from the broadcast variable
    if type(br_topics) == Broadcast:
       topics = br_topics.value
    else:
        topics = br_topics

    # only consider words that are at least in half of the topics for flood words
    flood_word_topics_cutoff = .50 * len(topics)
    print( flood_word_topics_cutoff )

    df = df.groupby(['word']).count()
    df = df.filter( F.col("count") >= flood_word_topics_cutoff )

    return df.select(F.col("word"))


class CoOccurrenceJob(BaseJob):

    def __init__(self):

        # invoke `BaseJob` constructor
        super().__init__(name="Co-Occurrence Analysis")

        # Support additional job-specific parameters
        # optionally aggregate topic counts to a time interval
        self.parser.add_argument(
            '--interval',
            choices=['daily', 'weekly', 'monthly', 'yearly', 'overall'],
            help="date interval for aggregations"
        )
        self.parser.add_argument(
            '--count_filter',
            default=25,
            type=int
        )
        self.parser.add_argument(
            '--word_length_filter',
            default=2,
            type=int
        )

        # either `topics` or `topic_file` can be provided
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--topics', help="JSON object from commandline")
        group.add_argument('--topics_file', help="JSON object from file")

        # adding default preprocessing_choices if none was passed in
        self.parser.set_defaults(preprocessing_choices=['lowercase', 'punctuation', 'stopwords', 'contractions'])
        # no need to tokenize in the script here since stopword will tokenize in the BaseJob

    def process(self):

        # construct output filename from parameters
        filename_components = [self.args.start_date, self.args.end_date,
                               self.args.dataset, "CoOccurrence", self.args.interval]
        output = construct_output_filename(None, self.args.output, filename_components)
        self.args.output = output + ".csv"

        print("output file: ", self.args.output)

        if self.args.topics:
            # Apache Airflow will convert double-quotes into single-quotes,
            # resulting in malformed JSON. We will convert these back to
            # double-quotes. This requires more research, as it is not an ideal
            # solution; it will convert single quotes within the string objects
            # themselves.
            if "\"" not in self.args.topics:
                self.args.topics = self.args.topics.replace("\'", "\"")
                topics = json.loads(self.args.topics)
        elif self.args.topics_file:
            # read topics file using spark to support reading from GCS
            topics = self.spark \
                .read \
                .option("multiline", True) \
                .json(self.args.topics_file) \
                .toPandas() \
                .to_dict(orient="records")

        # convert topic words to lower-case, and then broadcast to worker nodes
        for topic in topics:
            topic['words'] = [w.lower() for w in topic['words']]

        # construct dictionary of regular expressions to match topics
        global br_topics
        br_topics = self.sc.broadcast(topics)

        # this script needs tokenized text
        self.df = self.df.withColumn(
            "full_text_tokens",
            udf_tokenize(F.col(self.args.target_attr), F.lit(self.preprocess_lang))
        )

        udf_theme_proportions = F.udf(lambda r: topic_co_occurrence(r), TOPIC_SCHEMA)
        self.df = self.df.withColumn("topics", udf_theme_proportions(F.col("full_text_tokens")))

        # explode list of topics for each tweet into separate rows, this will also
        # filter out tweets that have no topics
        self.df = self.df.select(
            "date",
            F.explode("topics").alias("tt"),
        )

        # optionally aggregate to defined intervals
        if self.args.interval:

            if (self.args.interval == 'daily'):
                self.df = self.df.withColumn("interval", F.col("date"))
            elif (self.args.interval == 'weekly'):
                self.df = self.df.withColumn("interval", F.weekofyear(F.col("date")))
            elif (self.args.interval == 'monthly'):
                self.df = self.df.withColumn("interval", F.month(F.col("date")))
            elif (self.args.interval == 'yearly'):
                self.df = self.df.withColumn("interval", F.year(F.col("date")))
            elif (self.args.interval == 'overall'):
                self.df = self.df.withColumn("interval", F.lit("OVERALL"))

            # Extract the created date, topic, and word
            self.df.select("interval", "tt.topic", "tt.word")
            self.df = self.df.withColumn("word", F.explode("tt.word"))

            # Filter by word length
            self.df = self.df.filter(F.length(F.col("word")) > self.args.word_length_filter)

            # Group created date, topic, and word to count the number of words
            self.df = self.df.groupBy("interval", "tt.topic", "word").count()
            self.df = self.df.orderBy(["interval", "tt.topic", "word", "count"], ascending=[True, True, False, True])

        # no interval was passed in
        else:

            # Extract the created date, topic, and word
            self.df.select("date", "tt.topic", "tt.word")
            self.df = self.df.withColumn("word", F.explode("tt.word"))

            # Filter by word length
            self.df = self.df.filter(F.length(F.col("word")) > self.args.word_length_filter)

            # Group created date, topic, and word to count the number of words
            self.df = self.df.groupBy("date", "tt.topic", "word").count()
            self.df = self.df.orderBy(["date", "tt.topic", "word", "count"], ascending=[True, True, False, True])

        # Filter count rows only if it is greater than 7
        self.df = self.df.filter(F.col("count") > self.args.count_filter)

        # create list of possible flood words
   #     self.df = flood_word_contenders(self.df)


if __name__ == "__main__":
    co = CoOccurrenceJob()
    co.run()
