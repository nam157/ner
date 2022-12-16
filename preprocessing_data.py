import ast
import json
import re
import string

import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader


def process_string(tweet):
    """
    - remove Emojis
    - remove punctuation
    - remove HTML tags,URL,hyperlinks
    """
    punc = """!()-[]{};:'"\,<>./?$%^&*_~"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE,
    )
    tweet = re.sub(r"https?://[^\s\n\r]+", "", tweet)
    tweet = emoji_pattern.sub(r"", tweet)
    #     tweet = contractions.fix(tweet)
    tweet = re.sub(r",", "", tweet)
    no_punct = [words for words in tweet if words not in punc]
    words_wo_punct = "".join(no_punct)
    string_list = words_wo_punct.split()
    return string_list


def create_list_collections(file="./config.yaml"):
    with open(file) as f:
        data = yaml.load(f, Loader=SafeLoader)
    name_collection = []
    for i in range(len(data["collections_and_keywords"])):
        name_collection.append(data["collections_and_keywords"][i]["name"])
        name_collection.extend(data["collections_and_keywords"][i]["keywords"])
    return name_collection


def map_keyword_2_collection(list_keyword):
    ls = []
    for i in list_keyword:
        ls.append(1 if i in name_collection else 0)
    return ls


if __name__ == "__main__":
    df = pd.read_csv("./tweets_collections.csv")
    # clean_data
    df["tweet_content"] = df["tweet_content"].apply(lambda x: process_string(x))
    # list collections_and_keywords
    name_collection = create_list_collections()
    # map labels {1: collections, 0: O}
    df["ner_tag"] = df["tweet_content"].apply(lambda x: map_keyword_2_collection(x))

    df = df.drop(["collection_address", "name", "slug", "keywords"], axis=1)
    df.to_csv("collections_content.csv", index=True)

    df_after = pd.read_csv("./collections_content.csv")
    df_after.columns = ["id", "tokens", "ner_tags"]
    df_after["tokens"] = df_after["tokens"].apply(lambda x: ast.literal_eval(x))
    df_after["ner_tags"] = df_after["ner_tags"].apply(lambda x: ast.literal_eval(x))
    df_after["id"] = df_after["id"].astype(str)

    dict_df = df.to_dict(orient="records")
    with open("datasets/train_ner.json", "w") as outfile:
        json.dump(dict_df, outfile)
