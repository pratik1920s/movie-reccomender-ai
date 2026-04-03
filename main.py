# Step 3: Load and merge data
import pandas as pd
import ast  # Needed later for JSON parsing

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on movie title
movies = movies.merge(credits, on='title')

# Check first 5 rows
print(movies.head())

# Step 4: Choose important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Step 5: Clean data

# Remove missing values
movies.dropna(inplace=True)

# Convert JSON-like columns into lists
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Keep top 3 cast members
def convert_cast(text):
    L = []
    count = 0
    for i in ast.literal_eval(text):
        if count < 3:
            L.append(i['name'])
            count += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Get director from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Check cleaned data
print(movies.head())
# Step 6: Remove spaces in names (so 'Sam Worthington' becomes 'SamWorthington')
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview (text) to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Step 6: Create 'tags' column by combining all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Step 7: Create new DataFrame with only required columns
new_df = movies[['movie_id','title','tags']]

# Convert list of words into single string for vectorization
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Step 7: Vectorize text using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Step 7: Calculate cosine similarity matrix
similarity = cosine_similarity(vectors)

print("Vectorization and similarity calculation done!")# Step 8: Recommendation function
def recommend(movie):
    if movie not in new_df['title'].values:
        print("Movie not found! Please check spelling.")
        return

    # Find the index of the movie
    index = new_df[new_df['title'] == movie].index[0]
    
    # Get similarity scores for all movies
    distances = similarity[index]
    
    # Sort movies by similarity (highest first) and get top 5
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("\nRecommended Movies:\n")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Step 8: Test the function
movie_name = input("Enter movie name: ")
recommend(movie_name)