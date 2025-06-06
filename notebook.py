# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: C:\Users\ASUS\AppData\Roaming\Python\share\jupyter\kernels\python3
# ---

# %% [markdown]
# ---
# title: 'Topic: <span style="color:red">Bitcoin StackExchange Discussion Forum</span>'
# jupyter: python3
# ---
#
# **Name:** Vu Hoang Nam Doan 
#
# **Student Number:** s224021565
#
# **Email:** s224021565@deakin.edu.au  
#
# **Course:** S379 - Bachelor of Data Science
#
# **Unit:** Undergraduate (SIT220)
#
# ## <span style="color:red">Data Privacy and Ethics Consideration:</span>
#
# This analysis uses publicly available data from Bitcoin StackExchange, where users share information openly. While the data is public, we focused on ethical practices by analyzing data in groups to respect user privacy. We looked at Bitcoin-related content, such as addresses and financial discussions, to identify trends without trying to identify individuals or promote any financial opinions. Our findings aim to be clear and transparent, while also recognizing that sensitive topics in cryptocurrency can be easily misunderstood.
#
#
# ## Abstract
#
# This report presents a comprehensive analysis of the Bitcoin StackExchange dataset, a public Q&A platform for Bitcoin enthusiasts, including approximately 32,000 questions and 44,000 answers. The primary objective of this report is to explore and understand the user base, the sentiment expressed in discussions, and the key topics that characterize this online community. Methodologies employed include data ingestion and preprocessing of the original XML data into Pandas DataFrames, followed by extensive Exploratory Data Analysis (EDA). Key analytical techniques applied are user distribution analysis (including geographical mapping and profile content examination), sentiment analysis of posts and comments using an enhanced VADER lexicon, tag co-occurrence network analysis, and Latent Dirichlet Allocation (LDA) for topic modeling and evolution. This study aims to provide insights into the thematic structure, community sentiment, and salient characteristics of discussions within the Bitcoin StackExchange forum, highlighting common user concerns and areas of technical focus.
#
# ## I. Introduction
#
# Online Question and Answer (Q&A) platforms, such as the Stack Exchange (SE) network, have become invaluable repositories of community-generated knowledge, reflecting the collective intelligence, evolving interests, and persistent challenges faced by their users. Within this network, specialized sites like Bitcoin StackExchange serve as critical hubs for developers, users, and enthusiasts to discuss the intricacies of Bitcoin, its underlying blockchain technology, and related software development. The textual data generated within these forums—comprising questions, answers, and comments—offers a rich, dynamic dataset for understanding the Bitcoin ecosystem from the perspective of its active participants.   
#
# This report outlines a comprehensive plan for conducting an Exploratory Data Analysis (EDA) and subsequent in-depth analysis of data extracted from the Bitcoin StackExchange platform. The primary objective is to systematically explore the content and dynamics of these discussions to uncover patterns, trends, and insights related to location, Bitcoin topic, and community sentiment. 
#
# The data set used in this project is a collection of Bitcoin discussion forum posts. The dataset is collected from the Bitcoin StackExchange website, which is a question-and-answer platform for Bitcoin enthusiasts. The dataset contains a total of 32k questions and 44k answers. In this report we will focus primarily on the text columns of the dataset, along with the combination with data, score, and view data columns. The first section will provide a guide to convert the XML files, which is the original format of the dataset, into CSV files, provide a guide to convert the CSV files into a Pandas DataFrame. The second section will introduce a list of necessary functions that we will need for the remaining analysis in the report. Afterward, the third section will provide parts for various analyses. First we analyze the user distribution analysis, which analyze the location and About Me informationof the user. Afterwards, we will work on posts and comments dataset to analyze the sentiment of the posts and comments. This analysis includes various exploratory data analysis techniques, like word cloud, and sentiment analysis. The last part of the EDA and Data Visualization section will focus on the tag and topic of the posts via the Latent Dirichlet Allocation (LDA) Model, performing various data visualization and insights.
#
# ## II. Data Ingestion and Preprocessing
#
# In this section, we will explore foundational procedures for preparing the Bitcoin StackExchange dataset for subsequent analysis. It commences with an exploration of the inherent StackExchange data schema, providing a structural overview of the dataset. Subsequently, the methodology for transforming the original XML data files into usable Pandas DataFrames is described; this involves an intermediary conversion to CSV format. Finally, the section introduces a compendium of essential custom functions developed to facilitate standardized data manipulation and analytical operations throughout this study.
#
# Before we begin, we must import all necessary Python libraries that will be utilized for the subsequent data analysis, visualization, and modeling tasks throughout this Notebook.
#
# - Standard library tools for operating system interactions and data collections
# - Data processing and analysis packages like NumPy and Pandas
# - XML parsing capabilities
# - Natural Language Processing (NLP) tools from NLTK and scikit-learn for tasks such as tokenization, lemmatization, sentiment analysis, and stop word removal
# - Libraries for data visualization like Matplotlib, Seaborn, and WordCloud
#
# Machine learning tasks will leverage scikit-learn for models Random Forest Regressor we will perform later. For network analysis and geospatial analysis, NetworkX, python-louvain, and Folium are imported, respectively. This comprehensive set of imports ensures that all required tools are readily available for the analyses presented in this report.

# %%
# Standard library imports
import os
import re
import glob
import itertools
from collections import Counter, defaultdict
from datetime import datetime

# Data processing and analysis
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

# Data parsing
import xml.etree.ElementTree as ET

# Natural Language Processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk.sentiment.vader as vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Geospatial analysis
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster

# %% [markdown]
# As we import the NLTK package, we must download essential resources from this library, which are required for various text processing tasks. 
# - `punkt` for sentence tokenization
# - `averaged_perceptron_tagger` for part-of-speech tagging
# - `wordnet` for lemmatization and lexical relations
# - `vader_lexicon` for sentiment analysis
#
# Since we only need to download these rersources once, we can comment out the download code. If you run this code for the first time, please uncomment the following lines to download the resources. 

# %%
# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# %% [markdown]
# ### A. StackExchange Data Schema Exploration
#
# The Bitcoin StackExchange dataset is a structured collection of user interactions and content generated within the platform. It is distributed as a set of interconnected data files, originally in XML format, detailing various aspects of the forum's activity. Understanding this schema is fundamental to comprehending the relationships between different entities and performing meaningful analyses. In this report, we will focus on three most important tables: `Users`, `Posts`, and `Comments`.
#
# - The `Users` table contains information about each registered user on the Bitcoin StackExchange forum. In the context of this analysis, we will focus on some key attributes, including Location and AboutMe for geospatial analysis and users' profile analysis.
#
# - The `Posts` table is central to the dataset, containing the textual content of questions and answers. It includes attributes such as PostTypeId (indicating whether the post is a question or an answer), CreationDate, Score, ViewCount, and Body (the main content of the post). In this analysis, we will focus on the Body column, which contains the text of the posts to analyze the sentiment of the posts. The PostTypeId column will be used to filter the posts into questions and answers for further tags analysis.
#
# - The `Comments` table stores comments made on posts (both questions and answers). The attributes include PostId (linking the comment to its respective post), Score, and Text (the content of the comment). The Text column will be used to analyze the sentiment of the comments, and further combined with the Posts table to analyze the sentiment of the posts and comments.
#
# The remaining tables provide additional context and metadata about the tags, votes, badges, post history, and links between posts.
#
# - The `Tags` table provides information about the tags used on the site.
#
# - The `Votes` table logs all votes cast on posts.
#
# - The `Badges` table contains records of badges awarded to users for various achievements.
#
# - The `PostHistory` table tracks the revisions and significant events for each post.
#
# - The `PostLinks` table contains information about the relationships between different posts, such as duplicate questions.
#
# ### B. XML to Pandas DataFrame Conversion
#
# After understanding the data schema, to effectively analyze the Bitcoin StackExchange data, we convert the original XML files into a more analysis-friendly format, which is CSV files. Subsequently, these CSV files are loaded into Pandas DataFrames, providing a structured and efficient way to manipulate and analyze the data.
#
# #### 1. Convert XML data files to CSV 
#
# We first define the function `xml_to_csv` to convert the XML files into CSV files. The function takes the input XML file path and the output CSV file path as arguments, then uses the `xml.etree.ElementTree` module to parse the XML file and extract the relevant data. The extracted data is then saved as a CSV file using the `pandas` library.

# %%
def xml_to_csv(xml_file):
    output_file = os.path.splitext(xml_file)[0] + '.csv'
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = [child.attrib for child in root]
    df = pd.DataFrame(data)
    
    df.to_csv(output_file, index=False)
    return df


# %% [markdown]
# Now we can define the function `process_xml_files` to process the XML files. This function takes the input directory containing the XML files and the output directory for the CSV files as arguments. In the function, it iterates through all XML files in the input directory, converts each file to CSV format using the `xml_to_csv` function, and saves the resulting CSV files in the output directory.

# %%
def process_xml_files(files):
    results = {}
    for xml_file in files:
        try:
            file_name = os.path.basename(xml_file)
            df = xml_to_csv(xml_file)
            results[file_name] = df
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    
    return results


# %% [markdown]
# The dictionary `xml_file_list` below contains the list of XML files for processing. We will apply the function `process_xml_files` to convert the XML files from this list into CSV files. The output CSV files will be saved in the same directory. 

# %%
xml_file_list = [
    'Badges.xml',
    'Comments.xml',
    'PostHistory.xml',
    'PostLinks.xml',
    'Posts.xml',
    'Tags.xml',
    'Users.xml',
    'Votes.xml',
]

# %%
data = process_xml_files(xml_file_list)

# %% [markdown]
# #### 2. Convert CSV data files to Pandas DataFrame 
#
# After converting the XML files to CSV format, we can load the CSV files into Pandas DataFrames for further analysis. The dictionary `csv_file_list` below contains the list of CSV files after saved above.

# %%
csv_file_list = [
    'Badges.csv',
    'Comments.csv',
    'PostHistory.csv',
    'PostLinks.csv',
    'Posts.csv',
    'Tags.csv',
    'Users.csv',
    'Votes.csv',
]


# %% [markdown]
# We define the function `load_csv_files` to load the CSV files into Pandas DataFrames. This function takes the input directory containing the CSV files and returns a dictionary of DataFrames, where the keys are the file names and the values are the corresponding DataFrames.

# %%
def load_csv_files(file_list):
    dataframes = {}
    for file in file_list:
        try:
            df = pd.read_csv(file)
            dataframes[file] = df
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    return dataframes


# %%
csv_data = load_csv_files(csv_file_list)

# %% [markdown]
# #### 3. Assign DataFrame to specific variables
#
# We assign each element in the dictionary `csv_data` containing the DataFrames we read earier to specific variables for easier access.

# %%
badges_df = csv_data['Badges.csv']
comments_df = csv_data['Comments.csv']
post_history_df = csv_data['PostHistory.csv']
post_links_df = csv_data['PostLinks.csv']
posts_df = csv_data['Posts.csv']
tags_df = csv_data['Tags.csv']
users_df = csv_data['Users.csv']
votes_df = csv_data['Votes.csv']


# %% [markdown]
# ### C. Necessary Functions for Data Analysis
#
# For more efficient data analysis, we define a set of functions that will be used throughout the report. In thi section, we have a initial set of 14 general functions that will be used for data analysis, visualization, and modeling tasks. They include functions for data cleaning, text processing, sentiment analysis, and visualization. These functions are designed to be reusable and modular, allowing for easy integration into various parts of the analysis.
#
# #### 1. Extracting tags from the posts
#
# This function is designed to extract tags from the posts in the DataFrame.

# %%
def extract_tags(text):
    if pd.isna(text):
        return []
    tags = text.split('|')
    return [tag.strip() for tag in tags if tag.strip()]


# %% [markdown]
# #### 2. Cleaning the location data
#
# This function is designed to clean the location data in the DataFrame.

# %%
def clean_users_location(users_df):
    df = users_df.copy()
    df['Location'] = df['Location'].fillna('Unknown')
    return df


# %% [markdown]
# #### 3. Converting the date column to datetime format
#
# This function convert the date column to datetime format.

# %%
def convert_date_columns(df, date_columns):
    df_copy = df.copy()
    for col in date_columns:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy


# %% [markdown]
# #### 4. Removing the html tags from the text column
#
# This function removes the html tags from the text column in the DataFrame.

# %%
def remove_html_tags(text):
    html_pattern = re.compile(r'<.*?>')
    clean_text = re.sub(html_pattern, '', str(text))
    return clean_text


# %% [markdown]
# #### 5. Cleaning special characters from the text column
#
# This function is designed to clean the text data in the DataFrame. The function uses regular expressions to remove special characters, URLs, and emojis from the text. It also replaces certain keywords with more general terms. For example, it replaces "btc" and "xbt" with "bitcoin", "eth" with "ethereum", and "mempool", "utxo", and "segwit" with "technical_term". Additionally, it removes any extra whitespace, strips leading or trailing spaces, and removes the code blocks and inline code formatting. 

# %%
def clean_special_characters(text):
    clean_text = re.sub(r'```[\s\S]*?```', ' ', text)
    clean_text = re.sub(r'`[^`]*`', ' ', clean_text)
    clean_text = re.sub(r'https?://\S+|www\.\S+', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = re.sub(r'\b(?:quot|amp|lt|gt|apos|rsquo|lsquo)\b', '', clean_text)
    clean_text = re.sub(r'\b[b-hj-z]\b', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\b(?:btc|xbt)\b', 'bitcoin', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\b(?:eth)\b', 'ethereum', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\b(?:mempool|utxo|segwit)\b', 'technical_term', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\b(?:mining)\b', 'bitcoin mining', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\b(?:wallet)\b', 'bitcoin wallet', clean_text, flags=re.IGNORECASE)

    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0]+",
        flags=re.UNICODE
    )
    clean_text = re.sub(emoji_pattern, '', clean_text)

    return clean_text


# %% [markdown]
# #### 6. Get POS tags from the text column
#
# This function get the POS tags from the text column in the DataFrame. The POS tags are used to identify the parts of speech in the text, which can be useful when performing sentiment analysis or other text processing tasks.

# %%
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# %% [markdown]
# #### 7. Lemmatizing the text column
#
# This function lemmatizes the text column in the DataFrame. The lemmatization process reduces words to their base or root form, which can help improve the accuracy of text analysis tasks such as sentiment analysis and topic modeling.

# %%
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokens])


# %% [markdown]
# #### 8. Tokenizing the text column
#
# This function tokenizes the text column in the DataFrame, breaking down text into individual words or tokens, which can be useful for various text analysis tasks, including sentiment analysis and topic modeling.

# %%
def tokenize_text(text):
    tokens = text.split()
    filtered_tokens = [
        word for word in tokens 
        if word.lower() 
        not in ENGLISH_STOP_WORDS and len(word) > 1
    ]
    
    return filtered_tokens


# %% [markdown]
# #### 9. Lowercasing the text column
#
# This function lowercases the text column in the DataFrame. Lowercasing is a common preprocessing step in text analysis, as it helps to standardize the text and reduce the number of unique tokens, making it easier to analyze and visualize the data.

# %%
def lowercase(text):
    return text.lower()


# %% [markdown]
# #### 10. Calculating the sentiment score of the text column
#
# Because Bitcoin is a financial topic, it can contain some specific lexicons that are not included in the VADER lexicon. Therefore, we have to define the function `enhance_vader_for_crypto` to enhance the VADER lexicon for Bitcoin. For instance, the word "hodl" is a popular term in the Bitcoin community, which means to hold onto your Bitcoin instead of selling it. We set the sentiment score of "hodl" to 2.0, which means it has a positive sentiment.

# %%
def enhance_vader_for_crypto():
    sid = SentimentIntensityAnalyzer()
    
    crypto_lexicon = {
        "hodl": 2.0, 
        "moon": 3.0,
        "dump": -2.0,
        "fud": -2.5,
        "bullish": 2.5,
        "bearish": -2.5,
        "whale": 0.5,
        "hack": -3.0,
        "fork": -0.5,
        "halving": 1.5,
        "adoption": 2.0,
        "scam": -3.0,
        "regulation": -1.0,
        "btfd": 2.0,
        "fomo": -1.0
    }

    sid.lexicon.update(crypto_lexicon)    
    return sid


# %%
crypto_sid = enhance_vader_for_crypto()


# %% [markdown]
# The function `polarity_score` now use the enhanced VADER lexicon to calculate the sentiment score of the text column in the DataFrame.

# %%
def polarity_scores(text):
    scores = crypto_sid.polarity_scores(text)
    return scores['compound']


# %% [markdown]
# #### 11. Determining the sentiment of the text column
#
# This function compare the computed sentiment score with a threshold to determine the sentiment of the text.

# %%
def bitcoin_sentiment_analysis(text):
    compound = polarity_scores(text)
    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


# %% [markdown]
# #### 12. Generating the word cloud
#
# This function create a word cloud from the text column in the DataFrame. The word cloud visually represents the most frequently occurring words in the text, with larger words indicating higher frequency. This can be useful for quickly identifying key themes and topics within the text data.

# %%
def create_word_cloud(text, title, ax):
    wordcloud = WordCloud(
        background_color='white',
        width=1000,
        height=600
    )

    wordcloud.generate(text)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')


# %% [markdown]
# #### 13. Get lat and long from the location column
#
# This function return the latitude and longitude of the location column in the DataFrame. The function uses the `geopy` library to geocode the location data, which converts the location names into geographic coordinates. This can be useful for visualizing the data on a map or performing geospatial analysis.

# %%
def get_lat_long(location: str):
    if not location or not isinstance(location, str):
        return None, None
    if location in _cache:
        return _cache[location]
    
    result = geocode(location, timeout=10)
    coords = (result.latitude, result.longitude) if result else (None, None)
    _cache[location] = coords
    
    return coords


# %% [markdown]
# #### 14. Extracting the country from the location column
#
# This function extracts the country from the location column in the DataFrame. The function uses the `geopy` library to reverse geocode the latitude and longitude coordinates, which converts the geographic coordinates back into a human-readable location. This can be useful for analyzing the distribution of users by country or region.

# %%
def extract_country(location):
    geolocator = Nominatim(user_agent="BitcoinStackExchange_Analysis")
    result = geolocator.reverse((location[0], location[1]), exactly_one=True, language='en')
    if result and 'address' in result.raw:
        return result.raw['address'].get('country')
    return None


# %% [markdown]
# ## III. EDA & Data Visualization
#
# ### A. User Distribution Analysis
#
# In this section, we will investigate the characteristics of the Bitcoin StackExchange user base, focusing on their geographical distribution and the content of their profiles in the About Me section to gain insights into the community's composition and self-representation.

# %%
users_subset = users_df.dropna(
    subset=[
        "Location",
        "AboutMe"
    ])[['Id', 'Location', 'AboutMe']].copy()

# %% [markdown]
# #### 1. Geographical Mapping of User Base
#
# To understand the global reach and concentration of the Bitcoin StackExchange community, we will focus on analyzing the self-reported location data provided by users in this section, culminating in a visual representation of their geographical distribution.
#
# ##### a. Data Preparation and Exploratory Data Analysis
#
# First, we will clean the location data in the Users DataFrame using the function `clean_special_characters`. This function will remove any unnecessary characters and standardize the location format. 

# %%
users_subset['Location'] = users_subset['Location'].apply(clean_special_characters)

# %% [markdown]
# We will extract the unique values from the cleaned location for getting the longitude and latitude of the location.

# %%
print(f"\nUnique location entries: {users_subset['Location'].nunique()}")

# %% [markdown]
# Because the Nominatim API has a limit of 1 request per second, which can lead to a long wait time for geocoding a large number of locations, we will use the `RateLimiter` from the `ratelimiter` library to limit the number of requests to 1 per second. We will also use the `swallow_exceptions` parameter to ignore any exceptions that occur during the geocoding process. This will allow us to continue processing the remaining locations even if some of them fail to geocode.

# %%
geolocator = Nominatim(user_agent="BitcoinStackExchange_Analysis")
geocode = RateLimiter(
    geolocator.geocode, 
    min_delay_seconds=0.1, 
    max_retries=0, 
    swallow_exceptions=True
)
_cache = {}

# %% [markdown]
# Now we can extract the Location column from the Users DataFrame and apply the `get_lat_long` function to get the latitude and longitude of the location.

# %%
unique_locs = users_subset['Location'].dropna().unique()
for loc in unique_locs:
    get_lat_long(loc)

# %% [markdown]
# After getting the latitude and longitude of each location, we will store the results in new two columns `Latitude`, and `Longitude`.

# %%
users_subset[['Latitude', 'Longtitude']] = (
    users_subset['Location']
    .map(lambda x: _cache.get(x, (None, None)))
    .tolist()
)

# %% [markdown]
# ##### b. Generate The Geographical Map for User Base
#
# Currently, we have the latitude and longitude of each user's location. We can use this information to create a geographical map of the user base. 
#
# To implement the map, we will use the `folium` library to create an interactive map that displays the locations of users based on their self-reported locations.

# %%
world_map = folium.Map(location=[20, 0], zoom_start=2)

marker_cluster = MarkerCluster().add_to(world_map)

for idx, row in users_subset[users_subset['Latitude'].notna()].iterrows():
    popup_text = f"User: {row.name}<br>Location: {row['Location']}"
    folium.Marker(
        location=[row['Latitude'], row['Longtitude']],
        popup=popup_text,
        tooltip=row['Location']
    ).add_to(marker_cluster)

world_map

# %% [markdown]
# The geographic distribution of Bitcoin StackExchange users reveals a striking bimodal concentration in North America and Europe, each accounting for around one-third of all participants. In North America, the United States alone contributes over 5,800 users, clustered primarily along the East (4,037) and West (1,682) Coasts, while Canada has 125 users. Additionally, Europe mirrors this engagement, with Western and Central Europe combining for nearly 5,300 users (1,472 and 3,797, respectively) and less strong showings in the North-East area of 666. Therefore, these two regions represent over two-thirds of the forum’s active base.
#
# Besides, Asia has the largest non-Western stronghold, with India leading at approximately 1765 users, followed by communities in South East Asia at 511 users and China at 296 users to the East. Moreover, Africa seems to have the lowest user distribution, with the number of users around over 500, followed by Oceania, including Australia (around 430 users) and New Zealand (97 users), and South America with nearly 650 users.
#
# Overall, looking at the map, North America and Europe dominate the conversation, while India drives Asia’s participation, combined with China and Southeast Asia countries. Regions like Latin America, Africa, and Oceania have a smaller figure of users but could offer high-growth opportunities. Consequently, the Bitcoin StackExchange community has mostly Western users, with a significant presence in North America and Europe, while Asia is represented by a smaller but growing user base.
#
# Besides, there're also some regions having a extremely low number of users, commonly from small islands. There're also a few locations that are impossible to the context, such as Antarctica because it is uninhabited.
#
# #### 2. User Profile Content Analysis
#
# ##### a. Data Cleaning and Exploratory Data Analysis
#
# Beyond geographical data, in this section, we will delve into the textual self-descriptions provided by users in their "About Me" sections, aiming to identify common themes, linguistic patterns, and the general nature of how users present themselves within the Bitcoin StackExchange community.
#
# First, we will clean the AboutMe data in the Users DataFrame using 3 functions defined earlier: `clean_special_characters`, `remove_html_tags`, and `lemmatize_text`. These functions will remove any unnecessary characters, HTML tags, and lemmatize the text to its base form.
#
# After cleaning the data, we also create a new column `AboutMe_Length` to store the length of the AboutMe text.

# %%
users_subset['Clean_AboutMe'] = users_subset['AboutMe'].apply(remove_html_tags)
users_subset['Clean_AboutMe'] = users_subset['Clean_AboutMe'].apply(clean_special_characters)
users_subset['Clean_AboutMe'] = users_subset['Clean_AboutMe'].apply(lemmatize_text)

users_subset['AboutMe_Length'] = users_subset['Clean_AboutMe'].apply(lambda text : len(text))

# %% [markdown]
# We also apply the function `lowercase` to convert the text to lowercase.

# %%
users_subset['Clean_AboutMe'] = users_subset['Clean_AboutMe'].apply(lowercase)

# %% [markdown]
# ##### b. About Me Length Analysis
#
# Before delve into the textual content of the About Me section, we will first analyze the length of the text in this section. This analysis will help us understand how much information users typically provide about themselves.
#
# In order to analyze it, we will create a histogram to visualize the distribution of the length of the About Me text. The histogram will display the frequency of different lengths of text, allowing us to quickly identify the most common lengths and any outliers in the data. 
#
# Adjacent to the histogram, we will also create a boxplot to visualize the distribution of the length of the About Me text. The boxplot will provide a summary of the data, including the median, quartiles, and any potential outliers.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(
    users_subset['AboutMe_Length'], 
    bins=20, color='skyblue', edgecolor='black'
)
ax1.set_title('Distribution of About Me Text Length')
ax1.set_xlabel('About Me Text Length (characters)')
ax1.set_ylabel('Frequency')

mean_val = users_subset['AboutMe_Length'].mean()
ax1.axvline(
    mean_val, 
    linestyle='--', 
    label=f'Mean: {mean_val:.1f}'
)
ax1.legend()

ax2.boxplot(users_subset['AboutMe_Length'], vert=False)
ax2.set_title('Boxplot of About Me Text Length')
ax2.set_ylabel('About Me Text Length (characters)')

plt.tight_layout()
plt.show()

# %% [markdown]
# The generated histogram and boxplot provided a clear view of how users approach the "About Me" section. Looking at the histogram, the distribution of text length is highly right-skewed, meaning that while a few users write extensively, the majority keep their descriptions short. The histogram shows the highest frequency of entries at the lower character counts, typically under 250 characters, where the mean length of 183.8 characters is pulled higher by the longer texts. Additionally, the boxplot suggests a median length (the true 50th percentile) closer to 120-140 characters, showing that more than half of the users write less than this.
#
# Moreover, in the boxplot, the main box represents the middle 50% of users that is relatively compact and situated at the lower end of the scale. However, it also highlights numerous outliers representing those individuals who write significantly longer texts, one even exceeding 3000 characters. These outliers are responsible for the long tail in the histogram. 
#
# Overall, most users keep their "About Me" sections brief, while some users write more detailed descriptions of themselves, showing that they are more engaged or want to share more information.
#
# ##### c. Most Common Words in About Me Section
#
# In this section, we will analyze the most common words used in the About Me section of the users. This analysis will help us understand the common themes and topics that users discuss in their self-descriptions in the next section. 
#
# To implement this, we first join all the cleaned AboutMe text into a single string, then use the `tokenize_text` function to tokenize the text into individual words.
#
# Afterwards, we use the `Counter` class from the `collections` module to count the frequency of each word in the tokenized text. We will then extract the 20 most common words and their frequencies.
#
# Finally, we will create a DataFrame to store the most common words and their frequencies, and sort the DataFrame by frequency in descending order.

# %%
about_me_corpus = ' '.join(users_subset['Clean_AboutMe'].dropna())
tokens = tokenize_text(about_me_corpus)

word_freq = Counter(tokens)
most_common_words = word_freq.most_common(20)

word_freq_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency')

# %%
display(word_freq_df.head())

# %% [markdown]
# Now we have the table of the most common words and their frequencies, we can create a horizontal bar plot to visualize the most common words in the About Me section. The bar plot will display the frequency of each word, allowing us to quickly identify the most frequently used words in the self-descriptions of users.

# %%
plt.figure(figsize=(15, 6))
plt.barh(word_freq_df['Word'], word_freq_df['Frequency'])
plt.xlabel('Frequency')
plt.title('Most Common Words in About Me Section')

# %% [markdown]
# This analysis of common words in "About Me" sections reveals an overwhelming focus on the technology and software development industry. Words like "developer", "software", and "work" lead in frequency with significantly high counts from over 3500 to nearly 4000, clearly painting a picture of a user base heavily populated by tech professionals. Other less significant terms mentioning specific skills such as "code", "java", and "python" alongside general terms like "computer", "program", and "technology" helped confirming the observation. Based on the frequency of these words, it might be evident that the majority of users are either software developers or have a strong interest in technology and programming. Interestingly, the words "project" and "time" are particularly significant, highlighting an emphasis on specific accomplishments, portfolio items, or deliverables – a core aspect of work in the tech field.
#
# Beyond defining their technical roles, users frequently discuss their "experience", "learning", and use "year" to frame their professional tenure. Additionally, we also notice some personal sentiments that are evident with words like "love" and "like", indicating users' share about their passion and interests. 
#
# Overall, the analysis of the most common words in the "About Me" section provides a more precise picture of individuals defining themselves by their technical expertise, professional journey, and project-based achievements, likely on a platform dedicated to the tech community.
#
# To observe the distribution of all words in the About Me section, we will create a word cloud to visualize the most frequently occurring words.

# %%
plt.figure(figsize=(10, 6))
ax = plt.gca()
create_word_cloud(' '.join(tokens), "Words Cloud in User's About Me", ax)
plt.tight_layout()
plt.show()


# %% [markdown]
# Extending from the analysis of the horizontal bar charts, this word cloud provides a compelling visual affirmation of the dominant themes within users' "About Me" sections. The most prominent words we can notice are "work" and "use" with the biggest font size, indicating a detailed shared of experiences and practices. This offers an immediate visual underscoring of the pervasive focus on the technology and software development professions. The word cloud also highlights the presence of other significant terms like "developer", "software", "engineer", and "project", which are not only large but also closely interconnected, suggesting a strong emphasis on the software development process and project-based work.
#
# Beyond these largest terms, it also gives prominence to related vocabulary like development, data, time, and specific programming languages such as python and java. Furthermore, this visualization highlights action-oriented and role-defining terms like build, create, design, solution, and team, providing picture of users actively engaged in constructive, problem-solving, and collaborative roles. Moreover, there're terms like company, client, startup, and specific fields like blockchain, AI, and cloud, indicating a professional context and a focus on emerging technologies.
#
# Overall, the word cloud enriches the previous bar chart analysis by offering a more holistic and immediate visual summary, confirming our earlier conclusion that these "About Me" sections are spaces where users share their technical skills, professional experiences, project involvements, and passion within the technology landscape.
#
# ##### d. General Themes of User Profiles
#
# After having the most common words in the About Me section, we can analyze the general themes of user profiles, which will help us identify the major information about the users, like jobs, interests, and other relevant information.
#
# First we will define a function `extract_ngrams` to extract n-grams from the text. N-grams are contiguous sequences of n items from a given sample of text or speech. In this case, we will use bigrams (n=2) to extract pairs of words that frequently occur together in the text.

# %%
def extract_ngrams(text, n=2):
    tokens = tokenize_text(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]


# %% [markdown]
# Now we apply the `extract_ngrams` function to the cleaned AboutMe text to extract bigrams. We will create a new column `AboutMe_Bigrams` in the DataFrame to store the extracted bigrams.
#
# Similarly to the previous section, we will then use the `Counter` class from the `collections` module to count the frequency of each bigram in the extracted bigrams. We will extract the 15 most common bigrams and their frequencies.
#
# Finally, we will create a DataFrame to store the most common bigrams and their frequencies, and sort the DataFrame by frequency in descending order.

# %%
users_subset['AboutMe_Bigrams'] = users_subset['Clean_AboutMe'].apply(lambda x: extract_ngrams(x, 2))

all_bigrams = []
for bigram_list in users_subset['AboutMe_Bigrams']:
    all_bigrams.extend(bigram_list)

bigram_freq = Counter(all_bigrams)

print("\nTop 15 most common word pairs in AboutMe:")
for bigram, count in bigram_freq.most_common(15):
    print(f"'{bigram}': {count}")

# %% [markdown]
# The list of common word pairs strongly confirm the technology-centric nature of the "About Me" sections, with specific job titles like "software engineer", "software developer", and "web developer" appearing frequently together, alongside typical professional phrases such as "year experience", "computer science", and "open source". Most users seem to be software engineers or developers with the frequency of 1026 and 695, respecively.
#
# We can create a word cloud to visualize the most frequently occurring bigrams in the About Me section.

# %%
wc = WordCloud(width=800, height=400, background_color="white") \
       .generate_from_frequencies(bigram_freq)

plt.figure(figsize=(15, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Bigrams Cloud in User's About Me")
plt.show()

# %% [markdown]
# Most users identify as "software engineers", "software developers", or "web developers", where their primary role involves "software development", "coding", and creating new programs or web-based solutions. Based on their profiles, they typically work on "projects" such as building "web applications", developing with "data science" or "machine learning", and utilising various "technologies" like Python or Java.
#
# To identify the presence of Bitcoin-related terms in user profiles, we will create a list of relevant terms and then count their occurrences in the About Me section. This is because the Bitcoin StackExchange community is primarily focused on Bitcoin and related topics, and users may include these terms in their self-descriptions to indicate their interests or expertise. Identifying these terms can provide insights into the users' backgrounds and their engagement with the Bitcoin ecosystem.

# %%
bitcoin_terms = ['bitcoin', 'crypto', 'blockchain', 'mining', 'wallet', 'trader', 
                'investor', 'developer', 'programmer', 'engineer']

print("\nBitcoin-related terms in user profiles:")
for term in bitcoin_terms:
    term_count = sum(users_subset['Clean_AboutMe'].str.contains(r'\b' + term + r'\b', case=False, na=False))
    percentage = (term_count / len(users_subset)) * 100
    print(f"{term}: {term_count} users ({percentage:.2f}%)")

# %% [markdown]
# After identifying the relevant terms, we can wrap them in a dictionary to create a DataFrame that contains the relevant terms and their frequencies. This will allow us to easily visualize the presence of these terms in the About Me section. For example, the term "developer" may indicate that the user is a software developer or engineer, while the term "trader" may indicate that the user is involved in trading or investing in Bitcoin.
#
# We will create a dictionary `themes` that contains the relevant terms and their corresponding themes. The keys of the dictionary are the themes, and the values are lists of relevant terms associated with each theme.

# %%
themes = {
    'Developer': ['developer', 'programmer', 'engineer', 'code', 'software', 'github'],
    'Trader/Investor': ['trader', 'invest', 'trading', 'investor', 'market', 'finance'],
    'Academic': ['research', 'study', 'phd', 'professor', 'student', 'university'],
    'Business': ['business', 'entrepreneur', 'startup', 'founder', 'ceo', 'company'],
    'Technical Enthusiast': ['technology', 'tech', 'enthusiast', 'passionate', 'interested'],
    'Privacy Advocate': ['privacy', 'security', 'anonymous', 'freedom', 'liberty']
}

# %% [markdown]
# We then create a DataFrame to store the relevant terms and their frequencies, and sort the DataFrame by frequency in descending order. The DataFrame will contain two columns: `Theme`, `Count`, and `Percentage`, where `Theme` is the theme of the relevant terms, and `Frequency` is the frequency of the relevant terms in the About Me section.

# %%
theme_counts = {}
for theme, keywords in themes.items():
    pattern = '|'.join(keywords)
    theme_counts[theme] = sum(users_subset['Clean_AboutMe'].str.lower().str.contains(pattern, regex=True))

# %%
theme_df = pd.DataFrame({
    'Theme': list(theme_counts.keys()),
    'Count': list(theme_counts.values()),
    'Percentage': [count/len(users_subset)*100 for count in theme_counts.values()]
}).sort_values('Count', ascending=False)

display(theme_df)

# %% [markdown]
# To observe clearly the difference between the themes, we will create a bar plot to visualize the presence of Bitcoin-related terms in the About Me section.

# %%
plt.figure(figsize=(10, 6))
plt.bar(theme_df['Theme'], theme_df['Percentage'])

plt.title('Common Themes in User Profiles')
plt.xlabel('Theme')
plt.ylabel('Percentage of Users')

plt.tight_layout()
plt.show()

# %% [markdown]
# This chart offers a clear view of the users on this platform, categorized by common themes found in their "About Me" sections. Overall, this community is overwhelmingly driven by technology. "Developers", individuals who professionally build software and tech solutions, represent the largest group, accounting for over 40% of all users. The second largest group is "Technical Enthusiasts" (around 20%), who are deeply passionate about tech even if not their main job. 
#
# Following this strong tech contingent, "Business" professionals (approximately 10%) and those in "Academic" fields (around 9%) form the next most significant segments. Their presence suggests the forum is also a valuable space for entrepreneurs, managers, students, and researchers, likely exploring or applying technology within their domains. Smaller dedicated groups like "Traders/Investors" and "Privacy Advocates" highlight more specific interests that resonate with parts of the community.
#
# ### B. Content Sentiment Analysis
#
# This section aims to quantify and interpret the emotional tone expressed within the textual content of the Bitcoin StackExchange platform, providing insights into the prevailing attitudes and opinions within the community's discussions.
#
# #### 1. Separate Sentiment Analysis
#
# In this section, we focus on the necessary preprocessing of textual data from posts and comments to prepare it for sentiment scoring, followed by a distinct analysis of the sentiment polarity and intensity within each of these content types.
#
# ##### a. Post Sentiment Analysis
#
# In this section, we will embark on our sentiment analysis by concentrating on the textual content within the main body of the posts. Our primary aim here is to preprocess this text, then creating word clouds in each sentiment category.
#
# To begin, we perform the initial text preprocessing steps crucial for our analysis of the posts data. For the Body column, we apply a series of cleaning functions: `remove_html_tags` to strip away any HTML markup, `clean_special_characters` to clean specific symbols, and `lemmatize_text` to reduce words to their fundamental dictionary forms. Finally, after these cleaning stages, we calculate the length of the resulting 'Clean_Body' and store in a new column 'Body_Length'.

# %%
posts_sentiment = posts_df.copy()
posts_sentiment["Clean_Body"] = posts_sentiment["Body"].apply(remove_html_tags).apply(clean_special_characters)
posts_sentiment["Clean_Body"] = posts_sentiment["Clean_Body"].apply(lemmatize_text).apply(lowercase)
posts_sentiment["Body_Length"] = posts_sentiment["Clean_Body"].apply(len)

# %% [markdown]
# Next, we calculate numerical sentiment scores using the `polarity_scores` function and then categorize these into sentiment labels (e.g., positive, negative, neutral) via the `bitcoin_sentiment_analysis` function, storing both results in new respective columns.

# %%
posts_sentiment["Body_Sentiment_Score"] = posts_sentiment["Clean_Body"].apply(polarity_scores)
posts_sentiment["Body_Sentiment"] = posts_sentiment["Clean_Body"].apply(bitcoin_sentiment_analysis)

# %% [markdown]
# Rename the 'Clean_Body' column to 'Post_Body' to maintain consistency with the 'Comment_Body' column created during the comments analysis.

# %%
posts_sentiment = posts_sentiment.rename(columns={"Clean_Body": "Post_Body"})

# %% [markdown]
# We then split the Posts DataFrame into 3 separate DataFrames: `positive_posts`, `negative_posts`, and `neutral_posts`. Each DataFrame contains the posts with positive, negative, and neutral sentiment, respectively.

# %%
positive_posts = posts_sentiment[posts_sentiment['Body_Sentiment'] == 'positive']
neutral_posts = posts_sentiment[posts_sentiment['Body_Sentiment'] == 'neutral']
negative_posts = posts_sentiment[posts_sentiment['Body_Sentiment'] == 'negative']

print(f"\nPositive posts: {len(positive_posts)}")
print(f"Neutral posts: {len(neutral_posts)}")
print(f"Negative posts: {len(negative_posts)}")


# %% [markdown]
# As we can see, the majority of the posts are positive (43526), followed by negative (22931) and neutral (8505) posts. This indicates that the Bitcoin StackExchange community tends to express clearly positive or negative sentiments in their discussions, with a smaller proportion of posts reflecting neutral sentiment.
#
# Now we will create 3 word clouds to visualize what the users are talking about in the different post sentiments.
#
# First we deploy the function `combine_post_text` to combine the text of the posts in each sentiment category into a single string.

# %%
def combine_post_text(df):
    return ' '.join(df['Post_Body'].astype(str).tolist())

positive_post_corpus = combine_post_text(positive_posts)
neutral_post_corpus = combine_post_text(neutral_posts)
negative_post_corpus = combine_post_text(negative_posts)

# %% [markdown]
# Then use the `generate_wordcloud` function to create a word cloud for each sentiment category.

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

create_word_cloud(
    text=positive_post_corpus,
    title="Positive", 
    ax=axes[0]
)
create_word_cloud(
    text=neutral_post_corpus,
    title="Neutral", 
    ax=axes[1]
)
create_word_cloud(
    text=negative_post_corpus,
    title="Negative", 
    ax=axes[2]
)

plt.suptitle("Bitcoin Forum by Sentiment")
plt.tight_layout()
plt.show()

# %% [markdown]
# This set of three word clouds shows the language used in a Bitcoin forum, divided by the sentiment Positive, Neutral, and Negative expressed in the posts. In all three sentiments, key Bitcoin words like bitcoin, transaction, wallet, address, key, miner, use, and blockchain are consistently common. This is expected since these terms are essential for any discussion about Bitcoin.
#
# In the Positive sentiment cloud, these main Bitcoin terms often appear with words that express actions and hopes, like create, make, want, need, and send. This means positive discussions focus on successfully using Bitcoin, setting up wallets or mining, wanting features, or reaching specific goals in the Bitcoin world.
#
# The Neutral sentiment cloud mixes core Bitcoin words with terms like data, question, node, run, bitcoin core (a specific software client), information, and network. This word cloud suggests that neutral posts usually deal with factual questions, technical explanations, sharing information, and straightforward descriptions of how Bitcoin systems work.
#
# In the Negative sentiment cloud, the main Bitcoin words appear alongside terms that highlight problems or concerns. There're some noticeable words like problem, don't, doesn't, issue, lost, and fee (often debated) that might indicate discussions about users facing difficulties, frustrations with transactions or wallets, worries about security, or high costs.
#
# Overall, while the main topics of Bitcoin, wallets, and transactions stay the same, the words used change according to the sentiment. Positive posts emphasize utility and goals, neutral posts discuss information and mechanics, and negative posts express challenges and concerns users have about Bitcoin.
#
# ##### b. Comment Sentiment Analysis
#
# In this section, we extend our sentiment analysis to the comments associated with posts. Similar to the previous section, we aim to create word clouds in each sentiment category for the comments.
#
# To begin, we again apply a series of cleaning functions: `remove_html_tags`, `clean_special_characters`, and `lemmatize_text`. Finally, after these cleaning stages, we calculate the length of the resulting 'Clean_Text' and store in a new column 'Comment_Length'.

# %%
comments_sentiment = comments_df.copy()
comments_sentiment["Clean_Text"] = comments_sentiment["Text"].apply(remove_html_tags).apply(clean_special_characters)
comments_sentiment["Clean_Text"] = comments_sentiment["Clean_Text"].apply(lemmatize_text)

# %% [markdown]
# Then apply the `polarity_scores` function to calculate the sentiment scores for the comments, and then categorize these into sentiment labels (e.g., positive, negative, neutral) via the `bitcoin_sentiment_analysis` function, storing both results in new respective columns.

# %%
comments_sentiment["Comment_Sentiment_Score"] = comments_sentiment["Clean_Text"].apply(polarity_scores)
comments_sentiment["Comment_Sentiment"] = comments_sentiment["Clean_Text"].apply(bitcoin_sentiment_analysis)

# %%
comments_sentiment = comments_sentiment.rename(columns={"Clean_Text": "Comment_Text"})

# %% [markdown]
# Like the posts, we will split the Comments DataFrame into 3 separate DataFrames: `positive_comments`, `negative_comments`, and `neutral_comments`.

# %%
positive_comments = comments_sentiment[comments_sentiment['Comment_Sentiment'] == 'positive']
neutral_comments = comments_sentiment[comments_sentiment['Comment_Sentiment'] == 'neutral']
negative_comments = comments_sentiment[comments_sentiment['Comment_Sentiment'] == 'negative']

print(f"\nPositive comments: {len(positive_comments)}")
print(f"Neutral comments: {len(neutral_comments)}")
print(f"Negative comments: {len(negative_comments)}")


# %% [markdown]
# We can see that the majority of the comments are positive (40773), followed by negative (21854) and neutral (21722) comments. This indicates that the Bitcoin StackExchange community tends to express clearly positive sentiments in their discussions, similar to the posts. However, the number of neutral comments is significantly higher than the number of neutral posts, indicating that users tend to express their opinions more clearly in comments than in posts.

# %%
def combine_comment_text(df):
    return ' '.join(df['Comment_Text'].astype(str).tolist())

positive_comment_corpus = combine_comment_text(positive_comments)
neutral_comment_corpus = combine_comment_text(neutral_comments)
negative_comment_corpus = combine_comment_text(negative_comments)

# %% [markdown]
# After combining the text of the comments in each sentiment category into a single string, we will create 3 word clouds to visualize what the users are talking about in the different comment sentiments.

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

create_word_cloud(
    text=positive_comment_corpus,
    title="Positive", 
    ax=axes[0]
)
create_word_cloud(
    text=neutral_comment_corpus,
    title="Neutral", 
    ax=axes[1]
)
create_word_cloud(
    text=negative_comment_corpus,
    title="Negative", 
    ax=axes[2]
)

plt.suptitle("Comment by Sentiment")
plt.tight_layout()
plt.show()

# %% [markdown]
# Like the post sentiment analysis, these three word clouds highlight the key words found in Bitcoin forum comments, separated by positive, neutral, and negative sentiments. This visualizations helps us understand the distinct roles comments play in these discussions.
#
# In positive comments, the word thank is exceptionally large, appearing frequently alongside use, answer, and bitcoin. They strongly suggest that users often write positive comments to express gratitude, acknowledge helpful solutions and answers they've received, or show appreciation within the community.
#
# Neutral comments are overwhelmingly characterized by the word use, with bitcoin, address, and transaction also being very prominent. These words show that neutral comments serve a highly functional or informational purpose, which used for giving brief instructions, sharing useful links, asking for specific clarifications, or providing factual details quickly.
#
# In negative comments, there're some noticeable words that stand out for negative sentiment, such as block, error, problem, issue, and fee. This shows that negative comments are typically where users describe difficulties they are facing, report errors with transactions or software, or raise concerns about specific problems they've encountered while discussing with other users.
#
# #### 2. Comment, Answer, and Post Sentiment Analysis
#
# Building upon the individual assessments, we will explore and compare the sentiment characteristics across different forms of user contributions—comments, answers, and original posts—to identify any distinct emotional expressions or patterns associated with each type of interaction.
#
# ##### a. Data Preparation and Exploratory Data Analysis
#
# To implement this, we define questions, answers, and comments as the three main types of posts in the Bitcoin StackExchange community, assigning their source type accordingly and merging them into a single DataFrame. This allows us to analyze the sentiment of all three types of posts in a unified manner, facilitating a more comprehensive understanding of the community's emotional landscape.

# %%
questions = posts_sentiment[posts_sentiment['PostTypeId'] == 1].copy()
answers = posts_sentiment[posts_sentiment['PostTypeId'] == 2].copy()

questions['Source'] = 'question'
answers['Source'] = 'answer'
comments_sentiment['Source'] = 'comment'

post_cols = ['Id', 'Post_Body', 'CreationDate', 'Score', 'Source', 'Body_Sentiment_Score', 'Body_Sentiment']
comment_cols = ['Id', 'Comment_Text', 'CreationDate', 'Score', 'Source', 'Comment_Sentiment_Score', 'Comment_Sentiment']

unified_posts = pd.concat([
    questions[post_cols].rename(columns={'Post_Body': 'Body', 'Body_Sentiment_Score': 'Sentiment_Score', 'Body_Sentiment': 'Sentiment'}),
    answers[post_cols].rename(columns={'Post_Body': 'Body', 'Body_Sentiment_Score': 'Sentiment_Score', 'Body_Sentiment': 'Sentiment'}),
    comments_sentiment[comment_cols].rename(columns={'Comment_Text': 'Body', 'Comment_Sentiment_Score': 'Sentiment_Score', 'Comment_Sentiment': 'Sentiment'})
], ignore_index=True)

# %% [markdown]
# We apply `convert_to_datetime` function to convert the CreationDate column to datetime format, and `tokenize_text` function to tokenize the text column.

# %%
unified_posts = convert_date_columns(unified_posts, ['CreationDate'])
unified_posts['Tokens'] = unified_posts['Body'].apply(tokenize_text)

# %% [markdown]
# After that, we can use display function to overview the DataFrame.

# %%
display(unified_posts.head())

# %% [markdown]
# ##### b. Stacked Bar Plot of Sentiment Distribution in Each Post Type
#
# To understand the questions, answers, and comments in the DataFrame more clearly, we will create a stacked bar to observe the percentage of each sentiment in each type of post. The stacked bar plot will display the percentage of positive, negative, and neutral sentiments for each type of post, allowing us to quickly identify the sentiment distribution across different types of posts.

# %%
plt.figure(figsize=(12, 6))
sentiment_by_source = unified_posts.groupby(['Source', 'Sentiment']).size().unstack()
sentiment_by_source_pct = sentiment_by_source.div(sentiment_by_source.sum(axis=1), axis=0) * 100
sentiment_by_source_pct.plot(kind='bar', stacked=True)
plt.title('Sentiment Distribution Percentages by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# This stacked bar chart reveals the sentiments positive, neutral, or negative within "answers", "comments", and "questions" on what appears to be a forum like Bitcoin StackExchange. From the plot, the positive sentiment is the most common across all post types. This is a healthy sign for a Q&A platform, suggesting users often find helpful interactions and a generally constructive environment when discussing Bitcoin topics.
#
# Both "answers" and "questions" show a similar pattern with a large portion being positive (nearly 60%), but there"s also a notable amount of negative sentiment (around 30%). This proportion reflects the cases where users often post "questions" when they encounter Bitcoin-related problems (which are negative situations). While "answers" seems to provide positive solutions regularly, they also carry a negative sentiment. This is because there might be some conflict between the answer and the question, leading to a negative sentiment in discussions.
#
# Similarly, "Comments" present a similar pattern, but with a more balanced distribution of sentiments. The positive sentiment is slightly lower than in "answers" and "questions", while the neutral sentiment is higher and the negative sentiment is lower. This pattern indicates that comments on Bitcoin StackExchange might serve for factual clarifications, suggestions, or brief exchanges, rather than complex problem-solving or emotional expressions found in questions or answers. 
#
# #### 3. Sentiment Trends Over Time
#
# To understand the evolution of community attitudes, we will investigate temporal patterns in sentiment. We will analyze how the sentiment expressed in posts and comments on the Bitcoin StackExchange platform has changed over the duration of the dataset, potentially identifying shifts in opinion or emotional responses to specific periods or events.
#
# First, we extract the month from the CreationDate column and create a new column 'Month' to store the month of each post. We will then group the DataFrame by month and calculate the average sentiment score for each month.

# %%
unified_posts['Month'] = unified_posts['CreationDate'].dt.to_period('M')
sentiment_over_time = unified_posts.groupby('Month')['Sentiment_Score'].mean()

sentiment_over_time.index = sentiment_over_time.index.to_timestamp("M")

# %% [markdown]
# Now we can create a line plot to visualize the average sentiment score over time. We will create a 5x2 matrix to show the average sentiment score of each post type in 10 years from 2014 to 2023.

# %%
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

for i, year in enumerate(years):
    year_data = sentiment_over_time[sentiment_over_time.index.year == year]

    year_data.plot(ax=axes[i], kind='line')

    axes[i].set_title(f'Sentiment in {year}')
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Average Sentiment')
    axes[i].set_ylim(sentiment_over_time.min() - 0.1, sentiment_over_time.max() + 0.1)

plt.suptitle('Bitcoin Discussion Sentiment Trends by Year', fontsize=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# This series of ten line charts shows the average sentiment of Bitcoin discussions each month from 2014 to 2023.  Overall, the collective sentiments in these online conversations generally were mostly in positive range, indicating that discussions, on average, avoid swinging between extreme negativity and excitement. However, this sentiment was rarely steady and experienced noticeable increases and decreases within each year, especially showing considerable variation in 2019 and 2023.
#
# Beyond the overall trend, there're some noticeable points related to specific period's sentiment variation. The end of 2020 saw a notably rise in positive sentiment, reaching one of the highest average points in the decade, which often aligns with periods of good news or strong market performance for Bitcoin. In contrast, years like 2022 showed more volatility with sharper changes in sentiment, reflecting the more turbulent and challenging phases in the Bitcoin ecosystem during that time. These charts demonstrate that public sentiment in Bitcoin discussions is constantly changing, paralleling the exciting and sometimes unpredictable journey of Bitcoin over the past ten years.
#
# ### C. Tag and Topic Analysis
#
# In this section, we shift our focus to the thematic structure of the discussions within the Bitcoin StackExchange forum. We will first examine the explicit relationships between user-assigned tags by constructing and visualizing a tag co-occurrence network, and subsequently employ Latent Dirichlet Allocation (LDA) to uncover underlying latent topics, analyze their distribution, correlate them with existing tags, and explore their evolution over time.
#
# #### 1. Tag Co-occurrence Network
#
# In this first subsection, we investigate the relationships between the tags assigned by users to posts. Our work involves preparing the tag data, identifying the most frequently used tags, and then constructing a co-occurrence matrix to visualize which Bitcoin-related subjects are most commonly discussed in conjunction with one another.
#
# ##### a. Data Preparation and Exploratory Data Analysis
#
# To begin our tag analysis, we first process the 'Tags' column from our questions dataset. In the following code cell, we 
#
# - Create a working copy of the existing questions data
# - Apply the extract_tags function to convert the raw tag strings into lists of individual tags for each post
# - Aggregate these using Counter to identify and print the 10 most frequently occurring Bitcoin tags with the corresponding counts.

# %%
questions_post = questions.copy()

questions_post['Tag_List'] = questions_post['Tags'].apply(extract_tags)

tags_list = [tag for tags in questions_post['Tag_List'] for tag in tags]
tag_counts = Counter(tags_list)
top_tags = tag_counts.most_common(10)

print("\nTop 10 Bitcoin discussion tags:")
for tag, count in top_tags:
    print(f"{tag}: {count}")

# %% [markdown]
# The output reveals that the most prominent discussion tags on the Bitcoin StackExchange forum related to the core technical infrastructure and client software, such as "bitcoin-core," "transactions," "blockchain," "wallet," and "bitcoind". It indicates most people in this forum focus on understanding the fundamental workings of Bitcoin, its practical usage, and specific software implementations. Additionally, the frequent appearance of tags like "transaction-fees," "lightning-network," "security," and "private-key" highlights user interest in practical aspects like cost-efficiency, scalability solutions, and the security of Bitcoin assets management.
#
# Now we identify top most pairs of tags that frequently co-occur in the same posts. This is done by creating a co-occurrence matrix, which counts how many times each pair of tags appears together in the same post. The resulting matrix is then filtered to retain only the top 20 most frequently co-occurring tag pairs.

# %%
tag_pairs = Counter()

for tag_list in questions_post['Tag_List']:
    if len(tag_list) > 1:
        pairs = list(itertools.combinations(sorted(tag_list), 2))
        tag_pairs.update(pairs)

min_occurrences = 5
significant_pairs = {k: v for k, v in tag_pairs.items() if v >= min_occurrences}

print("\nTop 10 most common tag pairs:")
for (tag1, tag2), count in sorted(significant_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"'{tag1}' & '{tag2}': {count} occurrences")

# %% [markdown]
# The strong pairing of "bitcoin-core" with "bitcoind" (751 occurrences) shows the technical focus on the primary Bitcoin software client and its daemon. Similarly, the frequent co-occurrence of "blockchain" & "transactions" (476), and "bitcoin-core" with both "blockchain" (421) and "transactions" (336), highlights that discussions often delve into the fundamental mechanics of the Bitcoin protocol and its operational aspects. The connections between "transactions" and "wallet" (267), and "transaction-fees" and "transactions" (261) discuss practical user concerns around managing and transacting Bitcoin, including the fees.
#
# To further explore the characteristics of individual tags, we now calculate their connectivity within the co-occurrence network. We will count how many tags each tag is connected to, then creates a DataFrame showing each tag's overall frequency, its number of connections, and the ratio of connections to frequency.

# %%
tag_connections = defaultdict(int)

for (tag1, tag2), _ in significant_pairs.items():
    tag_connections[tag1] += 1
    tag_connections[tag2] += 1

tag_analysis = pd.DataFrame({
    'Tag': list(tag_connections.keys()),
    'Frequency': [tag_counts.get(tag, 0) for tag in tag_connections],
    'Connections': list(tag_connections.values())
})

tag_analysis['Connection_Ratio'] = tag_analysis['Connections'] / tag_analysis['Frequency']
tag_analysis = tag_analysis.sort_values('Frequency', ascending=False)

display(tag_analysis.head(10))

# %% [markdown]
# This table presents the top 10 most frequent tags alongside their total frequency, the number of unique significant tags they co-occur with ('Connections'), and the ratio of these connections to their frequency. We observe that high-frequency tags like "bitcoin-core" and "transactions" also tend to have the highest number of connections, showing a linear relationship between tag frequency and connectivity.
#
# We will create a scatter plot to visualize and confirm the relationship between tag frequency and connectivity.

# %%
plt.figure(figsize=(10, 6))
plt.scatter(
    tag_analysis['Frequency'],
    tag_analysis['Connections'],
    alpha=0.6
)
plt.title('Tag Frequency vs. Number of Connections')
plt.xlabel('Tag Frequency (Number of Posts)')
plt.ylabel('Number of Connections')
plt.tight_layout()
plt.show()

# %% [markdown]
# This scatter plot illustrates a clear positive relationship between the frequency of a tag (how often it appears in posts) and the number of unique significant tags it co-occurs with. The majority of tags have lower frequencies and fewer connections, while a smaller number of highly frequent tags also exhibit a high degree of connectivity, suggesting these popular tags serve as important factors in the discussion network.
#
# ##### b. Co-occurrence Matrix of Top Bitcoin Tags
#
# To prepare for visualizing the tag co-occurrence network, we first construct an adjacency matrix. We first select the top N=12 most common tags, then initializes an N x N matrix with zeros. It populates this matrix with the co-occurrence counts for each pair of these top tags, drawing from the significant_pairs we identified earlier.

# %%
top_n = 12
top_tags_list = [tag for tag,_ in tag_counts.most_common(top_n)]

adj_matrix = np.zeros((top_n, top_n))

for i, tag1 in enumerate(top_tags_list):
    for j, tag2 in enumerate(top_tags_list):
        if i != j:
            if (tag1, tag2) in significant_pairs:
                adj_matrix[i, j] = significant_pairs[(tag1, tag2)]
            elif (tag2, tag1) in significant_pairs:
                adj_matrix[i, j] = significant_pairs[(tag2, tag1)]

# %% [markdown]
# Now we can create the heatmap to visualize the co-occurrence matrix. The heatmap will display the co-occurrence counts between the top tags, allowing us to quickly identify which tags are most frequently discussed together.

# %%
plt.figure(figsize=(10, 8))
plt.imshow(adj_matrix, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Co-occurrence Count')
plt.title('Co-occurrence Matrix of Top Bitcoin Discussion Tags')
plt.xticks(range(len(top_tags_list)), top_tags_list, rotation=45, ha='right')
plt.yticks(range(len(top_tags_list)), top_tags_list)
plt.tight_layout()
plt.show()

# %% [markdown]
# This heatmap provides a quantitative visualization of the co-occurrence frequencies between the top 12 most prominent tags within the Bitcoin StackExchange discussions. The intensity of the color in each cell corresponds to the strength of association between a pair of tags, with darker shades indicating a higher number of posts where both tags appear together. This visual representation allows for an immediate understanding of the thematic landscape and the interconnectedness of key concepts within the community's discourse.
#
# Overall, several key observations can be retrieved from this heatmap. Notably, a very strong co-occurrence is demonstrated between "bitcoin-core" and "bitcoind" (over 700 posts), highlighting discussions focusing on the primary Bitcoin client and its associated daemon. Similarly, the pair "blockchain" and "transactions" the second highest degree of co-occurrence with around 500 posts. This pair underscores the foundational relationship between the underlying distributed ledger technology and its primary function of recording transactions. Additionally, there're some considerable pairings, such as "bitcoin-core" with "wallet," "transactions," and "blockchain". Besides strong co-occurrences on the top left corner, we also observe some interesting relationships scattered throughout the matrix. The strong connection between "bitcoind" and "json-rpc" (approximately 400 posts) indicates a specific area of discussion focused on how to interact with the Bitcoin daemon through programming. 
#
# Recognizing these connections is important to spot areas where community members have expertise, and could help us create educational content or improve how people search for information on the platform. The heatmap clearly shows the most helpful tag combinations, giving us insights into how different parts of Bitcoin technology and its use are related in community conversations. Overall, people often discuss about the daemon and the core client together, and they also talk about the blockchain and transactions a lot.
#
# #### 2. Applying LDA Model to Identify Topics
#
# Following our analysis of explicit tags, this subsection delves into uncovering latent thematic structures within the post content using Latent Dirichlet Allocation (LDA). We will 
#
# - Detail the application of the LDA model
# - Examine the resulting topic distributions
# - Explore the correlation between these machine-generated topics and tags
# - Explore how frequently these topics have evolved over time.
#
# ##### a. Applying LDA Model
#
# In this subsection, we will apply the Latent Dirichlet Allocation (LDA) model, a probabilistic approach for topic modeling to the processed textual content of the questions. Our goal is to uncover latent thematic structures within the discussions, allowing us to identify abstract topics that emerge from the collective discourse of the Bitcoin StackExchange community. For further details on the LDA model, please refer to the [LDA documentation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) and [LDA Video Tutorial](https://www.youtube.com/watch?v=1_jq_gWFUuQ&pp=ygUfTGF0ZW4gRGlyaWNobGV0IEFsbG9jYXRpb24gTERBIA%3D%3D).
#
# To prepare the data specifically for LDA, we first create a working copy of our questions dataset and ensure that posts with no body content are excluded. We then apply an additional preprocessing step, `preprocess_for_lda`, which refines the text by removing very short words and tokens consisting only of digits, as these are typically less informative for topic discovery.

# %%
questions_for_lda = questions_post.copy()
questions_for_lda = questions_for_lda.dropna(subset=['Post_Body'])
print(f"\nWorking with {len(questions_for_lda)} question posts for topic modeling")

def preprocess_for_lda(text):
    tokens = [token for token in text.split() if len(token) > 2 and not token.isdigit()]
    return ' '.join(tokens)

questions_for_lda['LDA_Text'] = questions_for_lda['Post_Body'].apply(preprocess_for_lda)

# %% [markdown]
# To convert the textual data into a numerical format suitable for the LDA model, we will now employ the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique. We first initialize a TfidfVectorizer that is configured to ignore terms that appear in more than 95% of documents or in fewer than 2 documents, uses a standard list of English stop words, and limits the vocabulary to the top 5000 features. This vectorizer is then fitted to our preprocessed LDA text, transforming it into a TF-IDF matrix.

# %%
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    stop_words='english',
    max_features=5000
)

# Transform corpus to TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(questions_for_lda['LDA_Text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(feature_names)}")

# %% [markdown]
# With the TF-IDF matrix prepared, we now train our final Latent Dirichlet Allocation (LDA) model. In the following code cell, we define the LatentDirichletAllocation model, specifying the optimal 10 topics, along with parameters for reproducibility and efficient online learning, and then fit this model to our TF-IDF matrix to identify latent topics within the question posts.

# %%
optimal_num_topics = 10

final_lda_model = LatentDirichletAllocation(
    n_components=optimal_num_topics,
    random_state=42,
    learning_method='online',
    max_iter=25,
    learning_decay=0.7
)

print("Fitting final LDA model...")
final_lda_output = final_lda_model.fit_transform(tfidf_matrix)
print("LDA model training complete.")


# %% [markdown]
# ##### b. Topic Distribution
#
# We're currently having the trained LDA model, now turn to interpreting the identified latent topics. In this subsection, we will examine the characteristic words that define each topic and then analyze how the question posts are distributed across these thematic topics.
#
# To facilitate the examination of the topics generated by our LDA model, we define the `display_topics` function that will take the trained LDA model and the vocabulary as input, extract the top N most significant words, here we consume it's 10, for each topic along with their weights.

# %%
def display_topics(lda_model, feature_names, top_words=10):
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_indices = topic.argsort()[:-top_words-1:-1]
        top_features = [feature_names[i] for i in top_features_indices]
        top_weights = [topic[i] for i in top_features_indices]
        
        topics.append({
            'Topic': topic_idx + 1,
            'Words': top_features,
            'Weights': top_weights
        })
        
        print(f"Topic {topic_idx + 1}: {', '.join(top_features)}")
    
    return pd.DataFrame(topics)


# %%
topic_term_df = display_topics(final_lda_model, feature_names)

# %% [markdown]
# The LDA model has successfully distilled the forum discussions into ten distinct topics, each characterized by a unique set of keywords. These topics clearly illustrate various facets of the Bitcoin ecosystem, including:
#
# - *Topics 2, 9, and 1*: Highly technical aspects like core blockchain operations, transaction scripting, and software development in Topics 2, 9, and 1
#
# - *Topics 6 and 4*: Practical user concerns involving wallets, addresses, and transaction fees
#
# - *Topic 8*: Economic activities such as trading on exchanges
#
# - *Topics 10 and 7*: Explorations of scalability solutions like the Lightning Network or discussions encompassing other cryptocurrencies.
#
#
# After applying the LDA model, we now process its output to assign topics to each question. We first determine the most probable (dominant) topic for each question post, then store its full topic probability distribution, and record the probability score of this dominant topic.

# %%
questions_for_lda['Dominant_Topic'] = final_lda_output.argmax(axis=1)
questions_for_lda['Topic_Distribution'] = list(final_lda_output)
questions_for_lda['Dominant_Topic_Score'] = final_lda_output.max(axis=1)

# %%
display(questions_for_lda[[
    'Post_Body', 
    'Dominant_Topic', 
    'Dominant_Topic_Score'
]].head(5))

# %% [markdown]
# To understand the prevalence of each identified theme, we now quantify the number of questions primarily associated with each topic.

# %%
topic_counts = questions_for_lda['Dominant_Topic'].value_counts().sort_index()
topic_counts_df = pd.DataFrame(topic_counts).reset_index().rename(columns={'count': 'Number of Questions', 'Dominant_Topic': 'Topic'})

display(topic_counts_df)

# %% [markdown]
# To visually represent how the question posts are distributed among the identified themes, we now generate a bar plot. This plot will display each topic number on the x-axis and the corresponding number of questions predominantly assigned to that topic on the y-axis, offering a clear overview of the prevalence of each topic within the dataset.

# %%
plt.figure(figsize=(12, 6))
bars = plt.bar(topic_counts.index + 1, topic_counts.values) 
plt.xlabel('Topic Number')
plt.ylabel('Number of Questions')
plt.title('Distribution of Questions Across Topics')
plt.xticks(topic_counts.index + 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# The bar plot indicates the frequency of posts across ten distinct topics. An analysis of this data reveals a highly uneven and skewed distribution of activity.
#
# Specifically, Topic 6 (with 18,052 posts) and Topic 2 (with 10,499 posts) stand out as exceptionally dominant in a visual representation, indicating these two areas account for the vast majority of posts. Topic 8 also represents a substantial volume of activity with 2,170 posts. In contrast, most other topics show minimal engagement; for instance, Topics 4, 3, 1, 7, and 10 each contain fewer than 20 posts, signifying very low activity. Topic 9 (181 posts) and Topic 5 (22 posts) show modest, intermediate levels.
#
# Overall, we can infer that users commonly discuss around Topic 6 and 2. Based on the top words appearing in each topic, Topic 6 focuses on the practical, everyday use of Bitcoin, which covers discussions about how to use different types of Bitcoin wallets (software, hardware, mobile). Following this, Topic 2 seems to center on the creation of new bitcoins (mining) and the overall health and functioning of the Bitcoin network. Additionally, users also have some discussion about trading on exchanges and financial aspects of Bitcoin (Topic 8), which is a significant aspect of the Bitcoin ecosystem. Consequently, the topics of the forum revolves around existing Bitcoin applications and its economic implications, rather than complex technical development, advanced scaling solutions, or dedicated security discussions.
#
# ##### c. Topic-Tag Correlation Matrix
#
# To further understand the relationship between the algorithmically generated topics and the user-defined tags, we will now construct and analyze a correlation matrix. This matrix will quantify the association strength between each LDA topic and the most prevalent tags, offering insights into how well the discovered latent themes align with the community's explicit categorization of posts.
#
# We first define a dictionary topic_interpretations mapping each topic index to a human-readable theme based on its most frequent words we observed earlier.

# %%
topic_interpretations = {
    0: "Bitcoin Development and Libraries",
    1: "Mining and Network Operations",
    2: "Mining Hardware and Alternative Coins",
    3: "Wallet Security and Privacy",
    4: "Protocol Specifications and Development",
    5: "Wallet Usage and Transactions",
    6: "Alternative Cryptocurrencies",
    7: "Trading and Economics",
    8: "Bitcoin Scripting and Transactions",
    9: "Lightning Network and Scaling Solutions"
}

interpretation_df = pd.DataFrame({
    'Topic': list(range(1, optimal_num_topics + 1)),
    'Interpretation': [topic_interpretations[i] for i in range(optimal_num_topics)]
})
display(interpretation_df)


# %% [markdown]
# To support the qualitative understanding of each topic, we define a function `get_representative_docs`. This function will allow us to retrieve the top N most representative question posts for any given topic, based on their dominant topic scores, displaying their content, score, and creation date for closer examination.

# %%
def get_representative_docs(topic_num, df=questions_for_lda, n=3):
    topic_docs = df[df['Dominant_Topic'] == topic_num]
    representative_docs = topic_docs.sort_values('Dominant_Topic_Score', ascending=False).head(n)
    return representative_docs[['Post_Body', 'Dominant_Topic_Score', 'CreationDate']]


# %% [markdown]
# To gain a clear and deep understanding of the themes captured by our LDA model, we now display representative posts for a selection of topics. For example, we will iterate through Topic 2: Mining and Network Operations, Topic 8: Trading and Economics, and Topic 10: Lightning Network and Scaling Solutions, retrieves the top three posts most strongly associated with each topic, and prints excerpts of their content along with their dominant topic scores.

# %%
# Show examples for selected topics
for topic_num in [1, 7, 9]:
    print(f"\nTopic {topic_num+1}: {topic_interpretations[topic_num]}")
    for i, (_, row) in enumerate(get_representative_docs(topic_num).iterrows()):
        print(f"Example {i+1} (Score: {row['Dominant_Topic_Score']:.4f}): " + 
              (row['Post_Body'][:200] + "..." if len(row['Post_Body']) > 200 else row['Post_Body']))

# %% [markdown]
# The representative posts generally validate the assigned topic interpretations. 
#
# - Examples for "Mining and Network Operations" clearly discuss mining mechanics and network fundamentals 
# - "Trading and Economics" contains questions about exchanges and transaction limits.
# - Posts for "Lightning Network and Scaling Solutions" include relevant queries on routing fees.
#
# To prepare for data visualization correlating our LDA topics with the user-defined tags, we first consolidate the necessary labels. We will generate a list of the top 10 most frequent tags found in the dataset and a separate list of 10 LDA topics.

# %%
top_tags_list = [tag for tag, _ in tag_counts.most_common(10)]
topic_list = list(topic_interpretations.values())

# %% [markdown]
# We now construct a matrix to quantify the correlation between our algorithmically derived LDA topics and the most frequently used user-defined tags.

# %%
topic_tag_matrix = np.zeros((len(topic_list), len(top_tags_list)))
for topic_index, topic in enumerate(topic_list):
    topic_posts = questions_for_lda[questions_for_lda['Dominant_Topic'] == topic_index]

    for index, tag in enumerate(top_tags_list):
        tag_count = topic_posts['Tag_List'].apply(lambda tags: tag in tags).sum()
        if len(topic_posts) > 0:
            topic_tag_matrix[topic_index, index] = tag_count / len(topic_posts)

# %% [markdown]
# We now generate a heatmap from the previously computed topic-tag matrix. This visualization will use color intensity to represent the strength of correlation, with the x-axis showing the top tags and the y-axis displaying our interpreted LDA topics, making it easier to identify which tags are most characteristic of each discovered theme.

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(
    topic_tag_matrix,  
    fmt=".2f",
    cmap="YlGnBu", 
    xticklabels=top_tags_list, 
    yticklabels=[f"Topic {i+1}: {topic_interpretations[i][:15]}..." for i in range(optimal_num_topics)]
)
plt.xticks(rotation=45, ha='right')
plt.title("Topic-Tag Correlation Matrix")
plt.xlabel("Tags")
plt.ylabel("Topics")
plt.tight_layout()
plt.show()

# %% [markdown]
# The Topic–Tag Correlation Matrix displays, for each LDA‐derived topic, the proportion of posts tagged with one of the ten most frequent tags. Several patterns stand out. 
#
# Topic 1 (Bitcoin Development and LibLightning Network and Scaling Solutionsraries) shows a 40% association with bitcoin-core, with secondary peaks in transactions (20%) and bitcoind (12%), confirming its focus on core‐software queries. 
#
# Additionally, Topic 10 (Lightning Network and Scaling Solutions) correlates strongly with lightning-network (12%) and moderately with blockchain and security (around 11%), reflecting network mechanics and secure deployment concerns. 
#
# The Mining and Network Operations cluster (Topic 2) peaks on blockchain (12%) and also features transactions (8%) and transaction-fees (5%), while Mining Hardware and Alternative Coins (Topic 3) is more clear, displaying associations with transactions, address (around 10%), and lightning-network (18%), suggesting hardware discussions often interlink with off‐chain topics. 
#
# Wallet Security and Privacy (Topic 4) and Wallet Usage and Transactions (Topic 6) both span bitcoin-core, transactions, blockchain, and wallet tags (from 10 to 17%), with security skewing toward protocol details and usage toward user‐experience. 
#
# Niche topics like Protocol Specifications and Development (Topic 5) and Trading and Economics (Topic 8) carry lower tag densities (<10%), underscoring their cross‐cutting nature. 
#
# Bitcoin Scripting and Transactions (Topic 9) aligns moderately with bitcoin-core, transactions, address, and private-key, consistent with scripting and key‐management inquiries.
#
# ##### d. Topic Evolution Over Time
#
# In this subsection, we will analyze how the topics identified by the LDA model have evolved over time. We will create a line plot to visualize the number of posts associated with each topic over the years, allowing us to observe trends and shifts in community interests.
#
# To prepare the data for visualization, we will first extract the year from the CreationDate column and create a new column 'Year' to store the year of each post. We will then group the DataFrame by year and topic, counting the number of posts associated with each topic for each year.

# %%
questions_for_lda['Year'] = pd.to_datetime(questions_for_lda['CreationDate']).dt.year

# %%
topic_by_year = questions_for_lda.groupby('Year')['Dominant_Topic'].value_counts().unstack()
topic_by_year.columns = [f"Topic {i+1}" for i in range(optimal_num_topics)]
topic_by_year.index.name = 'Year'

# %% [markdown]
# After that, we can create a line plot to visualize the number of posts associated with each topic over the years. The x-axis will represent the years, while the y-axis will show the number of posts associated with each topic.
# This line plot illustrates the evolution of topics over time, revealing significant trends and shifts in community interests.

# %%
# Visualize topic trends
plt.figure(figsize=(16, 10))
ax = topic_by_year.plot(kind='line', marker='o')
plt.title('Evolution of Bitcoin Discussion Topics Over Time')
plt.xlabel('Year')
plt.ylabel('Total Number of Questions')
plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Bitcoin community discussions have been dominated by Wallet Usage and Transactions (Topic 6) and Mining and Network Operations (Topic 2), rising from low levels in 2011 to peaks around 2017 and declining toward the end of the period. Wallet discussions surged to 3,800 posts in 2017 before tapering under 1,000 by 2023. Mining and Network Operations topics mirrored this trend, peaking nearly 1600 posts, then receding to 950 and 500. Overall, most topics peaked around 2017, coinciding with Bitcoin's price surge and mainstream adoption, then dropped off, indicating a shift in community focus. 
#
# ## IV. Conclusion
#
# Throughout this report, we provided a comprehensive analysis of the Bitcoin StackExchange dataset, focusing on understanding the community's user base, the sentiment expressed in discussions, and the primary topics of interest. Through a structured process of data ingestion, careful preprocessing (including the conversion from XML to Pandas DataFrames), and the application of various exploratory data analysis techniques, we have extracted several key insights from this specialized online forum.
#
# Our analysis of user distribution revealed a global presence of participants, with noticeable concentrations in specific countries (North America and Europe), which we visualized through geographical mapping. The examination of user profiles, particularly the "About Me" sections, highlighted common themes related to software development and technical expertise. However, the majority of users provided only brief self-descriptions.
#
# Subsequently, we delved into sentiment analysis, which leveraging an enhanced VADER lexicon tailored for cryptocurrency discussions, indicated that while individual posts and comments vary, the overall tone of the platform tends to be slightly positive. With word clouds illustrating the language used in different sentiments, we can look at what users are talking about in positive, neutral, and negative posts. Additionally, we observed trends and differences in sentiment across questions, answers, and comments over time.
#
# Finally, the exploration of tags and topics focuses on the core concerns and organization of knowledge within the community. Analysis of tag co-occurrences identified strong relationships between fundamental concepts such as "bitcoin-core", "transactions", "blockchain", and "wallet", leading to a central component of the discussion. Furthermore, we also applied Latent Dirichlet Allocation (LDA) successfully to uncover distinct latent topics, including "Mining and Network Operations", "Wallet Security and Privacy", "Trading and Economics", "Bitcoin Scripting", and "Lightning Network and Scaling Solutions". The correlation of these LDA topics with user-defined tags provided validation and a deeper understanding of the thematic landscape.
#
# In conclusion, this study demonstrates the value of combining data processing mostly on text columns, natural language processing, and exploratory visualization techniques to discover meaningful patterns from the online community Bitcoin StackExchange. The findings provided insights into the community's interests and concerns, highlighting the dynamic nature of discussions within the forum. Moreover, future work could expand this analysis by incorporating advanced predictive modeling, exploring the evolution of specific technical solutions discussed, or comparing these findings with those from other cryptocurrency-focused forums.
#
