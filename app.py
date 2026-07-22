import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Netflix Global Analytics & ML",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Global Content Analytics & ML Dashboard")
st.write("An interactive analytics dashboard built to explore content trends, demographics, and real-time recommendations.")

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    
    # Cleaning missing values
    df['director'] = df['director'].fillna('Unknown Director')
    df['country'] = df['country'].fillna('Unknown Country')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['rating'] = df['rating'].fillna('NR')
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    
    # Date processing
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    
    # Duration parsing (e.g., '90 min' -> 90)
    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
    
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Global Filters")

# Content Type Filter
content_type = st.sidebar.multiselect(
    "Select Content Type:",
    options=df['type'].unique(),
    default=df['type'].unique()
)

# Release Year Filter
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
selected_years = st.sidebar.slider(
    "Select Release Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(2000, max_year)
)

# Apply Filters
filtered_df = df[
    (df['type'].isin(content_type)) &
    (df['release_year'] >= selected_years[0]) &
    (df['release_year'] <= selected_years[1])
]

# --- TOP METRIC CARDS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Titles", len(filtered_df))
m2.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
m3.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))
m4.metric("Countries Represented", filtered_df['country'].nunique())

st.markdown("---")

# --- ROW 1: CHARTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Content Type Distribution")
    fig_type = px.pie(
        filtered_df, 
        names='type', 
        title="Movies vs TV Shows Ratio", 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_type, use_container_width=True)

with col2:
    st.subheader("🌍 Top 10 Content Producing Countries")
    top_countries = filtered_df['country'].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    fig_country = px.bar(
        top_countries, 
        x='Country', 
        y='Count', 
        title="Production Count by Country",
        color='Count',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_country, use_container_width=True)

# --- ROW 2: TIME TRENDS & GENRES ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Content Added Over Time")
    timeline_df = filtered_df.groupby('year_added').size().reset_index(name='Count')
    fig_timeline = px.line(
        timeline_df, 
        x='year_added', 
        y='Count', 
        title="Cumulative Content Growth by Year",
        markers=True
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col4:
    st.subheader("🎭 Top 10 Genres (Using Explode Method)")
    # Splitting comma-separated genres
    exploded_genres = filtered_df.assign(listed_in=filtered_df['listed_in'].str.split(', ')).explode('listed_in')
    top_genres = exploded_genres['listed_in'].value_counts().head(10).reset_index()
    top_genres.columns = ['Genre', 'Count']
    fig_genres = px.bar(
        top_genres, 
        x='Count', 
        y='Genre', 
        orientation='h',
        title="Most Frequent Genres",
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig_genres.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_genres, use_container_width=True)

# --- ROW 3: AGE RATINGS DISTRIBUTION ---
st.subheader("🏷️ Content Classification by Age Rating")
rating_df = filtered_df['rating'].value_counts().reset_index()
rating_df.columns = ['Rating', 'Count']
fig_rating = px.bar(
    rating_df, 
    x='Rating', 
    y='Count', 
    title="Ratings Distribution (e.g., TV-MA, PG-13)",
    color='Rating'
)
st.plotly_chart(fig_rating, use_container_width=True)

# --- MACHINE LEARNING RECOMMENDATION SECTION ---
st.markdown("---")
st.subheader("🧠 Machine Learning: Content-Based Recommender (TF-IDF + Cosine Similarity)")

# Feature preparation
df['combined_features'] = df['listed_in'] + " " + df['country'] + " " + df['description']

@st.cache_resource
def build_ml_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_ml_model()

selected_movie = st.selectbox("Select a title to discover similar content:", df['title'].values)

if st.button("Get Recommendations"):
    idx = df[df['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Extract top 5 matches
    
    movie_indices = [i[0] for i in sim_scores]
    
    st.write(f"### Top 5 Recommendations for '{selected_movie}':")
    st.dataframe(
        df[['title', 'type', 'listed_in', 'country', 'release_year']].iloc[movie_indices],
        use_container_width=True
    )