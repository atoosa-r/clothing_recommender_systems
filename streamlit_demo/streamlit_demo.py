"""
# Recommender Systems Demonstration with Streamlit

Welcome to the Recommender Systems demonstration! This application showcases how advanced recommendation models work, focusing on a BERT-based or language-based recommender system. This tool is designed to help users understand the capabilities of our recommendation models in a user-friendly interface.

## Purpose
This demonstration allows users to experience the core functionality of our recommender systems without requiring prior purchase history. By entering a single sentence, users can see how the BERT-based system provides personalized product recommendations based on the input, simulating a customer's shopping history.

## How It Works
1. **Initial Overview 
        **Select an Index Group**: Choose a category to see the top words associated with it.
2. **Explore Recommender Systems**: Learn about different recommendation techniques.
3. **Get Recommendations**: Enter a sentence describing what you're looking for to receive tailored product suggestions.

Jump in and discover how our recommender systems can make your shopping journey more enjoyable and personalized!

"""

# Importing necessary libraries: 
import streamlit as st
import joblib 
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import torch

from PIL import Image
import os
import base64

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Background colour styling
page_style = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #EAEBE8;
        }
        [data-testid="stHeader"] {
            background-color: #EAEBE8;
        }
    </style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Box styling
box_style = """
    border: 2px solid #E51A44;
    padding: 20px;
    border-radius: 10px;
    margin: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    display: inline-block;
    width: 100%;
    text-align: center;
"""

# Title
title_html = f"""
<div style="{box_style}">
    <h3 style="color: black; font-weight: bold;"><strong>Enhancing Clothing Recommendations through Data-Driven Recommender Systems</strong></h3>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)
st.markdown("")

# Section: Why?
st.markdown("<h3 style='text-align: left;'><strong>Why?</strong></h3>", unsafe_allow_html=True)
st.markdown("""
- Tailoring the shopping experience to individual customer styles and preferences.
- Helping customers navigate large volumes of content or products.
- Tailored recommendations have the potential to boost customer retention rates by 25-35%.
- Recommender systems can increase the variety of items customers see by 40%.
""")

# Section: Solution and Impact
st.markdown("<h3 style='text-align: left;'><strong>Solution and Impact</strong></h3>", unsafe_allow_html=True)
st.markdown("""
- More personalized recommendations based on previous shopping history and similarity in shopping patterns.
- Enhanced customer satisfaction, experience, and reduced returns.
""")

# Divider
st.markdown("<hr style='border: none; height: 2px; background-color: #E51A44;' />", unsafe_allow_html=True)
st.markdown("")

# Section: Top Words by Index Group
st.markdown("<h3 style='text-align: left;'><strong>Top Words by Index Group</strong></h3>", unsafe_allow_html=True)
st.markdown("")

# Function to plot top words: 
def plot_top_words(group, title):
    all_words = ' '.join(group).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)

    words, counts = zip(*common_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#EAEBE8')
    ax.set_facecolor('#EAEBE8')

    ax.bar(words, counts, color='#E51A44')
    ax.set_title(title)
    ax.set_xlabel('\nWords')
    ax.set_ylabel('Frequency\n')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45)

    st.pyplot(fig)
    plt.clf()

# Load data function
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data
file_path = '~/Desktop/streamlit/cleaned_articles_dataframe.csv'
articles_df = load_data(file_path)

# User input for selecting index group
index_groups = articles_df['index_group_name'].unique()
selected_group = st.selectbox('Select Index Group', index_groups)

if selected_group:
    group = articles_df[articles_df['index_group_name'] == selected_group]['preprocessed_detail_desc']
    plot_top_words(group, f'Index Group: {selected_group}')

st.markdown("")
st.markdown("")

st.subheader("Recommender Systems:")

# Box styles for recommender systems
box_style = """
    border: 2px solid #E51A44;
    padding: 20px;
    border-radius: 10px;
    margin: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    display: inline-block;
    width: 30%;
    vertical-align: top;
"""

# Description content for each recommender system
user_item_recommender = f"""
<div style="{box_style}">
    <h4 style="color: #E51A44;">User-Item Recommender:</h4>
    <ul>
        <li>Making recommendations based on customers who share similar tastes.</li>
        <li>Customer A and B have high similarities.</li>
        <li>Customer A: 👕👕👖 👟👚</li>
        <li>Customer B: 👕👕👖</li>
        <li>Recommendation for Customer B: 👟👚</li>
    </ul>
</div>
"""

matrix_factorization_recommender = f"""
<div style="{box_style}">
    <h4 style="color: #E51A44;">Matrix Factorization Recommender:</h4>
    <ul>
        <li>Decomposing user-item interactions into latent factors.</li>
        <li>User A and Item X have similar latent factors.</li>
        <li>Latent Factors of Customer B: 🔵🔵🔴</li>
        <li>Latent Factors of Item X: 🔵🔵🔴</li>
        <li>Recommendation for User A: 📦 (Item X)</li>
    </ul>
</div>
"""

bert_based_recommender = f"""
<div style="{box_style}">
     <h4 style="color: #E51A44;">BERT-based Recommender:</h4>
    <ul>
        <li>Embedding detailed article descriptions using BERT.</li>
        <li>Embedding customer's previous shopping history.</li>
        <li>Customer C’s Shopping History: 👜👗👠</li>
        <li>Similar Items based on embeddings: 👜👗👠👒👚</li>
        <li>Recommendation for Customer C: 👒👚</li>
    </ul>
</div>
"""

# Showing each recommender system side by side
st.markdown(f"""
    <div style="display: flex; justify-content: space-between;">
        {user_item_recommender}
        {matrix_factorization_recommender}
        {bert_based_recommender}
    </div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Divider
st.markdown("<hr style='border: none; height: 2px; background-color: #E51A44;' />", unsafe_allow_html=True)
st.markdown("")

# Paths to load model and data
model_path = '~/Desktop/streamlit/BERT_model'
embeddings_path = '~/Desktop/streamlit/article_embeddings.npy'
ids_path = '~/Desktop/streamlit/article_ids.npy'
articles_path = '~/Desktop/streamlit/filtered_articles.pkl'
images_folder_path = '~/Desktop/streamlit/resized_images_sixty'

# Loading the BERT model
bert = SentenceTransformer(model_path)

# Loading article embeddings and IDs
article_embeddings = np.load(embeddings_path)
article_ids = np.load(ids_path)

with open(articles_path, 'rb') as f:
    filtered_articles = pickle.load(f)

# Function to get image path
def get_image_path(article_id, images_folder_path):
    article_id_str = str(article_id).zfill(9)
    folder_name = f"0{article_id_str[:2]}"
    file_name = f"0{article_id_str}.jpg"
    image_path = os.path.join(images_folder_path, folder_name, file_name)
    return image_path

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to recommend items
def recommend_items(input_sentence, top_n=5):
    input_embedding = bert.encode([input_sentence])[0]
    similarities = cosine_similarity([input_embedding], article_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_article_ids = [str(article_ids[i]).replace(',', '') for i in top_indices]
    top_similarities = similarities[top_indices]

    recommendations = pd.DataFrame({
        'article_id': top_article_ids,
        'similarity_score': top_similarities
    })

    return recommendations

# Styling for image container
st.markdown(
    """
    <style>
    .img-container {
        display: inline-block;
        margin: 5px;
        padding: 5px;
        border: 2px solid white;
        border-radius: 5px;
    }
    .img-container img {
        width: 100px;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# BERT-based recommender system section
st.markdown("<h2 style='text-align: left;'><strong>BERT-Based Recommender System</strong></h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left;'><strong>What are you shopping for today?</strong></h4>", unsafe_allow_html=True)
st.markdown("**Enter a sentence:** \n (For more accurate recommendations, please include a colour, department, style, or pattern)")
user_input = st.text_input("")

if user_input:
    top_recommendations = recommend_items(user_input, top_n=10)
    st.write("**Top Recommendations:**")
    
    cols = st.columns(5)
    for index, row in top_recommendations.iterrows():
        col = cols[index % 5]
        with col:
            st.write(f"Article ID: {row['article_id']}")
            st.write(f"Similarity Score: {row['similarity_score']:.4f}")
            image_path = get_image_path(row['article_id'], images_folder_path)
            if os.path.exists(image_path):
                img_base64 = image_to_base64(image_path)
                st.markdown(f'<div class="img-container"><img src="data:image/jpeg;base64,{img_base64}" /></div>', unsafe_allow_html=True)
            else:
                st.write(f"Image not found for Article ID: {row['article_id']}")
