import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#change the file path to the location of the movies_metadata.csv file on your computer
file_path = r'C:\\Users\\jakob\\DAT1E22Bxd\\Semester4\\BI BusinessIntelligence\\Week1\\Data\\movies_metadata.csv'

#Read the first 10000 rows from the CSV file
data = pd.read_csv(file_path, nrows=10000, low_memory=False)

tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]

# Initialize your Streamlit app
def main():
    st.title("Movie Recommender System")

    # Dropdown to select a movie
    movie_list = data['title'].tolist()
    selected_movie = st.selectbox("Choose a movie you like", movie_list)

    # Button to get recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie)
        # Display the recommendations
        for i, title in enumerate(recommendations):
            st.write(f"{i+1}: {title}")

if __name__ == "__main__":
    main()
