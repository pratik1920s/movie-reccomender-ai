import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="CineMatch AI", page_icon="🍿", layout="wide")

# --- TMDB API Helper ---
def get_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8" 
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- 1. Data Processing ---
@st.cache_resource
def prepare_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    df.dropna(subset=['overview'], inplace=True)
    
    # Helper for raw display
    df['genres_list'] = df['genres'].apply(lambda x: ", ".join([i['name'] for i in ast.literal_eval(x)]))
    df['cast_list'] = df['cast'].apply(lambda x: ", ".join([i['name'] for i in ast.literal_eval(x)][:5]))
    
    # Cleaning for Vectorization
    def clean_data(text):
        return [i['name'].replace(" ", "") for i in ast.literal_eval(text)]
    
    df['tags'] = df['overview'].apply(lambda x: x.split()) + \
                 df['genres'].apply(clean_data) + \
                 df['keywords'].apply(clean_data) + \
                 df['cast'].apply(lambda x: clean_data(x)[:3])
                 
    df['tags_str'] = df['tags'].apply(lambda x: " ".join(x))
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags_str']).toarray()
    similarity = cosine_similarity(vectors)
    
    return df, similarity, cv, vectors

df, similarity, cv, vectors = prepare_data()

# --- 2. State Management ---
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = df.iloc[0].title
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 Home"

# Function to handle movie clicks
def select_movie(title):
    st.session_state.selected_movie = title
    st.session_state.active_tab = "ℹ️ Movie Details"

# --- 3. Sidebar Auth ---
with st.sidebar:
    st.title("👤 Account")
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    # ... (Login/Signup code from previous steps remains here)

# --- 4. Main Interface ---
# We use a radio or selectbox as a tab-switcher to allow programmatic switching
tabs = ["🏠 Home", "ℹ️ Movie Details", "✨ Vibe Search"]
active_tab = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.active_tab))

# --- TAB: HOME ---
if active_tab == "🏠 Home":
    st.title("🎬 Popular Recommendations")
    search_choice = st.selectbox("Pick a movie you liked:", df['title'].values)
    
    if st.button("Find Similar"):
        idx = df[df['title'] == search_choice].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:7]
        
        cols = st.columns(6)
        for i, val in enumerate(distances):
            m_title = df.iloc[val[0]].title
            with cols[i]:
                st.image(get_poster(df.iloc[val[0]].movie_id))
                # Clicking this button triggers the jump to the Details page
                st.button("View Info", key=m_title, on_click=select_movie, args=(m_title,))

# --- TAB: MOVIE DETAILS ---
elif active_tab == "ℹ️ Movie Details":
    st.title("📖 Movie Information")
    
    # Allow manual selection too
    current_movie = st.selectbox("Select Movie:", df['title'].values, 
                                 index=list(df['title'].values).index(st.session_state.selected_movie))
    
    movie_data = df[df['title'] == current_movie].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(get_poster(movie_data.movie_id), use_container_width=True)
    with col2:
        st.header(movie_data.title)
        st.write(f"**Genres:** {movie_data.genres_list}")
        st.write(f"**Cast:** {movie_data.cast_list}")
        st.markdown("---")
        st.write(f"**Storyline:**\n{movie_data.overview}")

# --- TAB: VIBE SEARCH ---
elif active_tab == "✨ Vibe Search":
    st.title("✨ Search by Vibe")
    query = st.text_input("What are you in the mood for?")
    if st.button("Search"):
        query_vec = cv.transform([query]).toarray()
        sim = cosine_similarity(query_vec, vectors)
        indices = sorted(list(enumerate(sim[0])), reverse=True, key=lambda x: x[1])[:6]
        
        cols = st.columns(6)
        for i, idx_tuple in enumerate(indices):
            m_title = df.iloc[idx_tuple[0]].title
            with cols[i]:
                st.image(get_poster(df.iloc[idx_tuple[0]].movie_id))
                st.button("View Info", key=f"vibe_{m_title}", on_click=select_movie, args=(m_title,))