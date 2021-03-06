import emoji
import functools
import operator
import urllib

# Regex module with Unicode support
import regex as re


def split(text):
    """
    Split text, and separate emojis

    :param text: target text to split
    """

    # Separate emojis
    t_emoji = emoji.get_emoji_regexp().split(text)

    # Separate words
    t_whitespace = [substr.split() for substr in t_emoji]

    # Flatten the list
    t_split = functools.reduce(operator.concat, t_whitespace)

    return t_split


def remove_punctuation(text, ignore_urls=True):
    """ Remove punctuation using unicode supported regex engine.

    By default, if `ignore_urls=True`, then punctuation will not be removed
    from URLs. For example, the colon, slashes, and period will not be removed
    from the following URL: https://t.co/e626KBczSM

    Reference: https://www.regular-expressions.info/unicode.html

    :param text: target text to remove punctuation
    :param ignore_urls: do not remove punctuation from URLs
    """

    if ignore_urls:
        # Remove all punctuation, except for that which occurs within a URL, by
        # using a back-reference (\1) in the `sub` command
        punc_pattern = r"(https?://\S*)|[\p{P}\p{S}](?<![@#])"
        return re.sub(punc_pattern, r"\1", text)
    else:
        # Remove all punctuation, except for hashtags, and @-symbols using a
        # negative look-behind
        punc_pattern = r"[\p{P}\p{S}](?<![@#\'])"
        return re.sub(punc_pattern, "", text)


def tweet_text_cleanup(text):
    """
    Convert `text` to lowercase, repair hashtags, and remove newline characters

    :param text: string to prepare for processing
    """
    text = text.replace("# ", "#")
    text = text.replace("\n", " ")
    return text


def url_decode(text):
    """Decode percent-encoded URLs

    :param text: string to decode
    """
    return urllib.parse.unquote(text)
