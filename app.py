import streamlit as st
import pandas as pd
import plotly.express as px

# Configure the Streamlit webpage window behavior and full screen width
st.set_page_config(page_title="🎬 Netflix Data Science Dashboard", layout="wide")

# Inject custom HTML/CSS into the app to apply the Netflix dark brand UI style
st.markdown("""
    <style>
    .main { background-color: #141414; color: #FFFFFF; }
    h1, h2, h3 { color: #E50914 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Cache data loading in memory to prevent reloading the CSV on every interaction
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("netflix_titles.csv")
    
    # Handle missing values in text features by filling them with a default string
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['country'] = df['country'].fillna('Unknown Country')
    
    # Drop records where vital timeline or duration parameters are blank
    df = df.dropna(subset=['date_added', 'rating', 'duration'])
    
    # Standardize string date strings and cast them into valid DateTime objects
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')
    
    # Extract calendar year values to use as a numerical slider filter parameter
    df['year_added'] = df['date_added'].dt.year
    
    # Extract the leading digits from duration strings to create numeric features
    df['duration_num'] = df['duration'].str.split(' ').str[0].astype(int)
    
    return df

# Execute the data pipeline function to populate the dataframe
df = load_and_clean_data()

# Render UI layout controls inside the left-hand sidebar container
st.sidebar.title("🛠️ Dashboard Controls")
st.sidebar.write("Filter the metrics instantly below:")

# Multi-select widget allowing users to filter dataset rows by content types
selected_types = st.sidebar.multiselect(
    "Select Content Type:",
    options=df['type'].unique(),
    default=df['type'].unique()
)

# Range slider widget determining the boundary years for historical data mapping
min_year, max_year = int(df['year_added'].min()), int(df['year_added'].max())
selected_years = st.sidebar.slider(
    "Select Timeline Range (Year Added):",
    min_value=min_year,
    max_value=max_year,
    value=(2010, max_year)
)

# Filter rows matching user selected input options across both dimensions
filtered_df = df[
    (df['type'].isin(selected_types)) & 
    (df['year_added'] >= selected_years[0]) & 
    (df['year_added'] <= selected_years[1])
]

# Render the application header layout and print active content item totals
st.title("🎬 Netflix Global Catalog Analytics Dashboard")
st.write(f"Displaying evaluation analytics for **{filtered_df.shape[0]}** catalog titles based on selected filters.")
st.markdown("---")

# Split the screen horizontally into two equal layout columns (Row 1)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Catalog Proportion: Movies vs TV Shows")
    # Aggregate content counts to compare product volume breakdown ratios
    content_counts = filtered_df['type'].value_counts().reset_index()
    content_counts.columns = ['Content Type', 'Total Count']
    
    fig_pie = px.pie(
        content_counts, names='Content Type', values='Total Count',
        hole=0.4, color_discrete_sequence=['#E50914', '#221F1F']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Content Volume Additions Over Time")
    # Group content items by release year and type to establish historical trends
    timeline_df = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='Count')
    
    fig_line = px.line(
        timeline_df, x='year_added', y='Count', color='type',
        color_discrete_sequence=['#E50914', '#CCCCCC']
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# Split the screen horizontally into two equal layout columns (Row 2)
col3, col4 = st.columns(2)

with col3:
    st.subheader("🗺️ Top 10 High-Production Countries")
    # Compute content counts per geographic origin and filter out unknown strings
    top_countries = filtered_df['country'].value_counts().reset_index()
    top_countries.columns = ['Country', 'Total Content']
    top_countries = top_countries[top_countries['Country'] != 'Unknown Country'].head(10)
    
    fig_country = px.bar(
        top_countries, x='Total Content', y='Country', orientation='h',
        color='Total Content', color_continuous_scale=['#221F1F', '#E50914']
    )
    fig_country.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_country, use_container_width=True)

with col4:
    st.subheader("🏷️ Top 10 Distribution Genres")
    # Split delimited text strings and explode elements to break out distinct genres
    genres_df = filtered_df['listed_in'].str.split(', ').explode()
    top_genres = genres_df.value_counts().reset_index().head(10)
    top_genres.columns = ['Genre', 'Count']
    
    fig_genres = px.bar(
        top_genres, x='Genre', y='Count',
        color='Count', color_continuous_scale=['#221F1F', '#E50914']
    )
    st.plotly_chart(fig_genres, use_container_width=True)

st.markdown("---")

# Split the screen horizontally into two equal layout columns (Row 3)
col5, col6 = st.columns(2)

with col5:
    st.subheader("⏱️ Feature Film Runtime Distribution (Movies)")
    # Extract only Movie rows to trace length variations using regular bin groups
    movies_sub = filtered_df[filtered_df['type'] == 'Movie']
    
    fig_movie = px.histogram(
        movies_sub, x='duration_num', nbins=40,
        color_discrete_sequence=['#E50914'], labels={'duration_num': 'Minutes'}
    )
    st.plotly_chart(fig_movie, use_container_width=True)

with col6:
    st.subheader("📺 Serialization Lifespan Distribution (TV Shows)")
    # Extract only TV Show rows and sort them numerically by season index limits
    tv_sub = filtered_df[filtered_df['type'] == 'TV Show']
    tv_seasons = tv_sub['duration_num'].value_counts().reset_index().sort_values(by='duration_num')
    tv_seasons.columns = ['Seasons', 'Number of Shows']
    
    fig_tv = px.bar(
        tv_seasons, x='Seasons', y='Number of Shows',
        color='Number of Shows', color_continuous_scale=['#221F1F', '#E50914']
    )
    fig_tv.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_tv, use_container_width=True)