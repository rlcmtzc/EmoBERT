# Each preprocessing step should be a function with the same interface. (data pandas dataframe)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from rich.progress import track
from typing import List
import re
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')
nltk.download('omw-1.4')

SMILEYS = {
    ":‑)": "smiley",
    ":-]": "smiley",
    ":-3": "smiley",
    ":->": "smiley",
    "8-)": "smiley",
    ":-}": "smiley",
    ":)": "smiley",
    ":]": "smiley",
    ":3": "smiley",
    ":>": "smiley",
    "8)": "smiley",
    ":}": "smiley",
    ":o)": "smiley",
    ":c)": "smiley",
    ":^)": "smiley",
    "=]": "smiley",
    "=)": "smiley",
    ":-))": "smiley",
    ":‑D": "smiley",
    "8‑D": "smiley",
    "x‑D": "smiley",
    "X‑D": "smiley",
    ":D": "smiley",
    "8D": "smiley",
    "xD": "smiley",
    "XD": "smiley",
    ":‑(": "sad",
    ":‑c": "sad",
    ":‑<": "sad",
    ":‑[": "sad",
    ":(": "sad",
    ":c": "sad",
    ":<": "sad",
    ":[": "sad",
    ":-||": "sad",
    ">:[": "sad",
    ":{": "sad",
    ":@": "sad",
    ">:(": "sad",
    ":'‑(": "sad",
    ":'(": "sad",
    ":‑P": "playful",
    "X‑P": "playful",
    "x‑p": "playful",
    ":‑p": "playful",
    ":‑Þ": "playful",
    ":‑þ": "playful",
    ":‑b": "playful",
    ":P": "playful",
    "XP": "playful",
    "xp": "playful",
    ":p": "playful",
    ":Þ": "playful",
    ":þ": "playful",
    ":b": "playful",
    "<3": "love"
    }

# Probably only remove stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"] not all...
def remove_stopwords(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    en_stops = set(stopwords.words('english'))
    en_stops.remove("not") # maybe remove all negativ thingis like hadn't hasn't....
    
    for index, _ in data_copy.iterrows():
        clean_tweet = []
        for word in data_copy.at[index, "tweet"].split(" "):
            if word in en_stops: continue
            clean_tweet.append(word)
        data_copy.at[index, "tweet"] = " ".join(clean_tweet)

    return data_copy

# IMPORTANT DO THIS AFTER REMOVING LINKS and before removing non ascii
def remove_punctuation(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    punctuation = set("!$%&'()*+,-./:;<=>?[\]^_`{|}~")
    punctuation.add("...")
    for index, _ in data_copy.iterrows():
        for punc in punctuation:
            data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace(punc, "")
        #print(data_copy.at[index, "tweet"])
    return data_copy

def stem_tweets(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    ps = PorterStemmer()
    for index, _ in data_copy.iterrows():
        clean_tweet = []
        for word in word_tokenize(data_copy.at[index, "tweet"]):
            clean_tweet.append(ps.stem(word))
        data_copy.at[index, "tweet"] = " ".join(clean_tweet)
    return data_copy

def lemmatize_tweets(data: pd.DataFrame) -> pd.DataFrame:
    # There are better approaches: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    data_copy = data.copy()
    wnl = WordNetLemmatizer()
    for index, _ in data_copy.iterrows():
        clean_tweet = []
        for word in word_tokenize(data_copy.at[index, "tweet"]):
            clean_tweet.append(wnl.lemmatize(word))
        data_copy.at[index, "tweet"] = " ".join(clean_tweet)
    return data_copy

# Freestanding @ is used as "at" for locations. So replace it with at
def replace_at_with_at(data: pd.DataFrame) -> pd.DataFrame: 
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace(" @ ", " at ")
    return data_copy

def remove_extra_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = " ".join(data_copy.at[index, "tweet"].split())
    return data_copy

def replace_smileys(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        for smiley, emote in SMILEYS.items():
            data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace(smiley, emote)
    return data_copy

def remove_hashtags(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace("#", "")
    return data_copy

def unify_apostrophes(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace('"', "'")
    return data_copy

# Remove multiple extra letters to a max of two  loove love
def replace_extra_letters(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    pattern = re.compile(r"(.)\1{2,}")
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = pattern.sub(r"\1\1", data_copy.at[index, "tweet"])
    return data_copy

def make_lowercase(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].lower()
    return data_copy

def remove_digits(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        # Use regex to remove digits
        data_copy.at[index, "tweet"] = re.sub(r'\d+', '', data_copy.at[index, "tweet"])
    return data_copy

# Do spell checking??? https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html

def remove_urls(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = re.sub(r'https\S+', '', data_copy.at[index, "tweet"])
        data_copy.at[index, "tweet"] = re.sub(r'http\S+', '', data_copy.at[index, "tweet"])
    return data_copy

def remove_non_english_characters(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = re.sub(r'[^\x00-\x7F]+', '', data_copy.at[index, "tweet"])
    return data_copy

def remove_whole_hashtags(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = re.sub(r"#[A-Za-z0-9_]+", '', data_copy.at[index, "tweet"])
    return data_copy

def remove_only_hashtags(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace("#", "")
    return data_copy

# Needed after removing non englisch characters becaus of chinese hashtags...
def remove_freestanding_hashtags(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace(" # ", "")
        data_copy.at[index, "tweet"] = data_copy.at[index, "tweet"].replace("# ", "")
    return data_copy

# SHOULD BE THE LAST STEP!
def remove_tweets_with_less_than_two_words(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    for index, _ in data_copy.iterrows():
        if len(data_copy.at[index, "tweet"].split()) <= 2:
            data_copy.drop(index, inplace=True)
    return data_copy


def apply_random_oversampling(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df_0 = input_df.loc[input_df["emoji_class"] == 0]
    input_df_1 = input_df.loc[input_df["emoji_class"] == 1]
    input_df_2 = input_df.loc[input_df["emoji_class"] == 2]
    input_df_3 = input_df.loc[input_df["emoji_class"] == 3]
    input_df_4 = input_df.loc[input_df["emoji_class"] == 4]
    input_df_5 = input_df.loc[input_df["emoji_class"] == 5]
    input_df_6 = input_df.loc[input_df["emoji_class"] == 6]

    classes = input_df.emoji_class.value_counts().to_dict()
    most = max(classes.values())
    print(most)
    #input_df_0_os = input_df_0.sample(n=most, random_state=42, replace=True)
    input_df_1_os = input_df_1.sample(n=most, random_state=42, replace=True)
    input_df_2_os = input_df_2.sample(n=most, random_state=42, replace=True)
    input_df_3_os = input_df_3.sample(n=most, random_state=42, replace=True)
    input_df_4_os = input_df_4.sample(n=most, random_state=42, replace=True)
    input_df_5_os = input_df_5.sample(n=most, random_state=42, replace=True)
    input_df_6_os = input_df_6.sample(n=most, random_state=42, replace=True)

    # print(input_df_1)
    # print(input_df_1_os.sort_index())
    return pd.concat([input_df_0, input_df_1_os, input_df_2_os, input_df_3_os, input_df_4_os, input_df_5_os, input_df_6_os], sort=False)

# Give a list of preprocessors to run
# def apply_preprocessing(data: pd.DataFrame, preprocessing_steps: list[callable]) -> pd.DataFrame:
def apply_preprocessing(data: pd.DataFrame, preprocessing_steps: List[callable]) -> pd.DataFrame:
    for step in track(preprocessing_steps, description=f"Preprocessing Data:"):
        # print function name
        print(f"Currently performing: {step.__name__.replace('_', ' ')}")
        data = step(data)
    print("Done preprocessing!")
    return data


function_mapping = {
    "remove_stopwords": remove_stopwords,
    "remove_punctuation": remove_punctuation,
    "stem_tweets": stem_tweets,
    "lemmatize_tweets": lemmatize_tweets,
    "replace_at_with_at": replace_at_with_at,
    "remove_extra_whitespace": remove_extra_whitespace,
    "replace_smileys": replace_smileys,
    "remove_hashtags": remove_hashtags,
    "unify_apostrophes": unify_apostrophes,
    "replace_extra_letters": replace_extra_letters,
    "make_lowercase": make_lowercase,
    "remove_digits": remove_digits,
    "remove_urls": remove_urls,
    "remove_non_english_characters": remove_non_english_characters,
    "remove_whole_hashtags": remove_whole_hashtags,
    "remove_only_hashtags": remove_only_hashtags,
    "remove_freestanding_hashtags": remove_freestanding_hashtags,
    "remove_tweets_with_less_than_two_words": remove_tweets_with_less_than_two_words,
    "apply_random_oversampling": apply_random_oversampling,
}