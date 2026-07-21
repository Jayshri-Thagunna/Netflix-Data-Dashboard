import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

st.set_page_config(page_title="Netflix Global Analytics & ML", layout="wide")
st.title("🎬 Netflix Global Analytics & Deep Learning Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['director'] = df['director'].fillna('Unknown Director')
    df['country'] = df['country'].fillna('Unknown Country')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['rating'] = df['rating'].fillna('NR')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Content")
content_type = st.sidebar.multiselect("Select Content Type:", df['type'].unique(), default=df['type'].unique())
filtered_df = df[df['type'].isin(content_type)]

st.write(f"Showing **{len(filtered_df)}** records based on sidebar filters.")

# Analytics Visualizations
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

# Deep Learning: Recurrent Neural Network (RNN)
st.markdown("---")
st.subheader("🧠 Deep Learning: Recurrent Neural Network (RNN)")

descriptions = df['description'].fillna('').values

# Text Tokenization and Padding
max_words = 1000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(descriptions)

sequences = tokenizer.texts_to_sequences(descriptions)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# RNN Model Definition
@st.cache_resource
def build_rnn_model():
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
        SimpleRNN(units=32, return_sequences=False),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

rnn_model = build_rnn_model()

# RNN User Interface
user_text = st.text_area("Enter a custom movie plot description to test the RNN:", 
                         "A brave detective tracks down a mysterious killer across a dark city.")

if st.button("Predict with RNN"):
    user_seq = tokenizer.texts_to_sequences([user_text])
    user_padded = pad_sequences(user_seq, maxlen=max_len, padding='post')
    
    prediction = rnn_model.predict(user_padded)[0][0]
    
    st.write(f"**RNN Sequence Score:** `{prediction:.4f}`")
    if prediction > 0.5:
        st.success("The RNN classifies this description as **High-Intensity Content (Drama / Action / Thriller)**.")
    else:
        st.info("The RNN classifies this description as **Light-Hearted / General Audience Content**.")

# Content-Based Recommendation Engine
st.markdown("---")
st.subheader("🎯 Content-Based Recommendation Engine")

selected_title = st.selectbox("Select a title to get recommendations:", df['title'].values)

if st.button("Get Recommendations"):
    target_row = df[df['title'] == selected_title].iloc[0]
    target_genres = set(str(target_row['listed_in']).split(', '))
    target_country = str(target_row['country'])
    
    recommendations = []
    
    for idx, row in df.iterrows():
        if row['title'] == selected_title:
            continue
        
        score = 0
        current_genres = set(str(row['listed_in']).split(', '))
        matching_genres = target_genres.intersection(current_genres)
        score += len(matching_genres) * 2
        
        if str(row['country']) == target_country and target_country != 'Unknown Country':
            score += 1
            
        if score > 0:
            recommendations.append({
                'Title': row['title'],
                'Type': row['type'],
                'Genres': row['listed_in'],
                'Country': row['country'],
                'Match Score': score
            })
            
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        top_5 = rec_df.sort_values(by='Match Score', ascending=False).head(5)
        st.write(f"### Top 5 Recommendations for '{selected_title}':")
        st.dataframe(top_5[['Title', 'Type', 'Genres', 'Country', 'Match Score']], use_container_width=True)
    else:
        st.write("No similar titles found with the current filter settings.")