# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using **Python**, **Scikit-learn**, and **Streamlit**. The application recommends similar movies using cosine similarity and fetches movie posters using the TMDB API.

## ✨ Features

- Content-based movie recommendation
- Cosine Similarity algorithm
- Movie poster integration using TMDB API
- Interactive Streamlit interface
- Fast movie search
- Metadata preprocessing using Pandas

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Requests
- TMDB API

## 📁 Project Structure

```
movie-recommendation-system/
│
├── app.py
├── build_models.py
├── requirements.txt
├── README.md
├── model/
├── data/
└── notebooks/
```

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/tejasvini2006/movie-recommendation-system.git
```

Navigate to the project

```bash
cd movie-recommendation-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

## 🧠 Working

1. Load movie and credits datasets.
2. Preprocess movie metadata.
3. Extract important features (genres, cast, crew, keywords).
4. Convert text into vectors using CountVectorizer.
5. Calculate cosine similarity.
6. Recommend the top 5 similar movies.
7. Display movie posters using the TMDB API.

## 📸 Demo

Add screenshots here.

Example:

```
images/home.png
images/recommendation.png
```

## 👩‍💻 Author

**Tejasvini Kanani**

If you found this project helpful, consider giving it a ⭐.