import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

st.set_page_config(page_title="Netflix Global Analytics & ML", layout="wide")
st.title("Netflix Global Analytics & Machine Learning Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['director'] = df['director'].fillna('Unknown Director')
    df['country'] = df['country'].fillna('Unknown Country')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['rating'] = df['rating'].fillna('NR')
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Content")
content_type = st.sidebar.multiselect("Select Content Type:", df['type'].unique(), default=df['type'].unique())
filtered_df = df[df['type'].isin(content_type)]

st.write(f"Showing **{len(filtered_df)}** records based on sidebar filters.")

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Content Type Distribution")
    fig_type = px.pie(filtered_df, names='type', title="Movies vs TV Shows", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_type, use_container_width=True)

with col2:
    st.subheader("Top 10 Countries")
    top_countries = filtered_df['country'].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    fig_country = px.bar(top_countries, x='Country', y='Count', title="Top Content Producing Countries", color='Count')
    st.plotly_chart(fig_country, use_container_width=True)

# Machine Learning: TF-IDF + Cosine Similarity Recommender
st.markdown("---")
st.subheader("Machine Learning: Content-Based Recommender (TF-IDF)")

# Combine features into text string
df['combined_features'] = df['listed_in'] + " " + df['country'] + " " + df['description']

@st.cache_resource
def build_ml_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_ml_model()

selected_movie = st.selectbox("Select a title to get ML-based recommendations:", df['title'].values)

if st.button("Get ML Recommendations"):
    idx = df[df['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Extract top 5
    
    movie_indices = [i[0] for i in sim_scores]
    st.write(f"### Top 5 Recommendations for '{selected_movie}':")
    st.dataframe(df[['title', 'type', 'listed_in', 'country']].iloc[movie_indices], use_container_width=True)