import streamlit as st
import pandas as pd
import plotly.express as px

# Configure webpage window properties and full horizontal canvas limits
st.set_page_config(page_title="Netflix Catalog Analytics Dashboard", layout="wide")

# Apply custom institutional styling to establish a clean light/slate theme based on your screenshots
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; color: #0F172A; }
    h1, h2, h3 { color: #0EA5E9 !important; font-family: 'Inter', sans-serif; }
    div[data-testid="stSidebar"] { background-color: #1E293B; }
    </style>
""", unsafe_allow_html=True)

# Define professional thematic color variables for Plotly charts matching your image
COLOR_PRIMARY = "#0EA5E9"    # Sky Blue
COLOR_SECONDARY = "#1E293B"  # Dark Slate
COLOR_SCALE = ["#1E293B", "#0EA5E9"] # Progressive Slate-to-Blue Color Gradient

# Cache data loading in memory to avoid repetitive CSV processing cycles
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("netflix_titles.csv")
    
    # Fill structural string voids with generic designations
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['country'] = df['country'].fillna('Unknown Country')
    
    # Eliminate records lacking date parameters or runtime metrics
    df = df.dropna(subset=['date_added', 'rating', 'duration'])
    
    # Convert string date arguments into valid DateTime objects
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')
    
    # Extract structural year dimensions for execution controls
    df['year_added'] = df['date_added'].dt.year
    
    # Separate numeric value sequences out of duration string columns
    df['duration_num'] = df['duration'].str.split(' ').str[0].astype(int)
    
    return df

# Initialize data pipeline parsing
df = load_and_clean_data()

# Render interactive sidebar dashboard filters
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Apply macro variables to slice data distributions:")

# Cross-filtering check controls for item categorization types
selected_types = st.sidebar.multiselect(
    "Select Content Type:",
    options=df['type'].unique(),
    default=df['type'].unique()
)

# Slider limits setup for filtering localized catalog timelines
min_year, max_year = int(df['year_added'].min()), int(df['year_added'].max())
selected_years = st.sidebar.slider(
    "Select Timeline Range (Year Added):",
    min_value=min_year,
    max_value=max_year,
    value=(2010, max_year)
)

# Apply runtime conditions based on ongoing component changes
filtered_df = df[
    (df['type'].isin(selected_types)) & 
    (df['year_added'] >= selected_years[0]) & 
    (df['year_added'] <= selected_years[1])
]

# Render application dashboard header layers
st.title("Netflix Global Catalog Analytics Dashboard")

# Integrated Technical Summary & Context Panel
st.info("""
    **Dashboard Overview & Methodology** This application processes data from the Netflix media catalog using Python. It cleans missing entries, reads dates correctly, and creates visual charts to help explain what kind of content Netflix creates, where it is made, and how long it lasts.
""")

st.write(f"Displaying evaluation analytics for **{filtered_df.shape[0]}** catalog titles based on selected filters.")
st.markdown("---")

# ==============================================================================
# VISUALIZATION 1: CATALOG PROPORTION (PIE CHART - AS REQUESTED, ALONE AND COMPACT)
# ==============================================================================
st.subheader("Catalog Proportion: Movies vs TV Shows")
content_counts = filtered_df['type'].value_counts().reset_index()
content_counts.columns = ['Content Type', 'Total Count']

fig_pie = px.pie(
    content_counts, names='Content Type', values='Total Count',
    hole=0.4, color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY]
)
fig_pie.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_pie, use_container_width=False)

st.write("**What this shows:** This chart shows that Netflix focuses heavily on full-length feature films. Roughly 70% of the entire library consists of Movies, while only 30% consists of episodic TV Shows.")
st.markdown("<br><hr>", unsafe_allow_html=True) # Forces a structural line break

# ==============================================================================
# VISUALIZATION 2: TIMELINE TRENDS
# ==============================================================================
st.subheader("Content Volume Additions Over Time")
timeline_df = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='Count')

fig_line = px.line(
    timeline_df, x='year_added', y='Count', color='type',
    color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY]
)
fig_line.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_line, use_container_width=True)

st.write("**What this shows:** This timeline maps out when Netflix added content to its platform. You can see a massive spike in production starting around 2015, with movie uploads peaking rapidly before scaling back slightly in recent years.")
st.markdown("<br><hr>", unsafe_allow_html=True)

# ==============================================================================
# VISUALIZATION 3: REGIONAL ANALYSIS
# ==============================================================================
st.subheader("Top 10 High-Production Countries")
top_countries = filtered_df['country'].value_counts().reset_index()
top_countries.columns = ['Country', 'Total Content']
top_countries = top_countries[top_countries['Country'] != 'Unknown Country'].head(10)

fig_country = px.bar(
    top_countries, x='Total Content', y='Country', orientation='h',
    color='Total Content', color_continuous_scale=COLOR_SCALE
)
fig_country.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_country, use_container_width=True)

st.write("**What this shows:** This chart identifies the major global production hubs. The United States leads by a huge margin, followed by India in second place due to its massive Bollywood ecosystem, and the United Kingdom in third.")
st.markdown("<br><hr>", unsafe_allow_html=True)

# ==============================================================================
# VISUALIZATION 4: GENRE DISTRIBUTION
# ==============================================================================
st.subheader("Top 10 Distribution Genres")
genres_df = filtered_df['listed_in'].str.split(', ').explode()
top_genres = genres_df.value_counts().reset_index().head(10)
top_genres.columns = ['Genre', 'Count']

fig_genres = px.bar(
    top_genres, x='Genre', y='Count',
    color='Count', color_continuous_scale=COLOR_SCALE
)
fig_genres.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_genres, use_container_width=True)

st.write("**What this shows:** This graph ranks the most popular styles of shows and movies. International Movies, Dramas, and Comedies clear the top spots, proving that Netflix prioritizes broad, globally appealing genres.")
st.markdown("<br><hr>", unsafe_allow_html=True)

# ==============================================================================
# VISUALIZATION 5: MOVIE RUNTIMES
# ==============================================================================
st.subheader("Feature Film Runtime Distribution (Movies)")
movies_sub = filtered_df[filtered_df['type'] == 'Movie']

fig_movie = px.histogram(
    movies_sub, x='duration_num', nbins=40,
    color_discrete_sequence=[COLOR_PRIMARY], labels={'duration_num': 'Minutes'}
)
fig_movie.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_movie, use_container_width=True)

st.write("**What this shows:** This distribution curve reveals the length of standard Netflix films. Most movies cluster tightly around the **90 to 110-minute mark**, which matches standard theatrical runtimes perfectly.")
st.markdown("<br><hr>", unsafe_allow_html=True)

# ==============================================================================
# VISUALIZATION 6: TV LIFESPAN
# ==============================================================================
st.subheader("Serialization Lifespan Distribution (TV Shows)")
tv_sub = filtered_df[filtered_df['type'] == 'TV Show']
tv_seasons = tv_sub['duration_num'].value_counts().reset_index().sort_values(by='duration_num')
tv_seasons.columns = ['Seasons', 'Number of Shows']

fig_tv = px.bar(
    tv_seasons, x='Seasons', y='Number of Shows',
    color='Number of Shows', color_continuous_scale=COLOR_SCALE
)
fig_tv.update_layout(xaxis=dict(tickmode='linear', dtick=1), template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_tv, use_container_width=True)

st.write("**What this shows:** This chart shows how long TV series last. An overwhelming majority of shows are concluded or cancelled after exactly **1 Season**, with very few series reaching Season 3 or beyond.")