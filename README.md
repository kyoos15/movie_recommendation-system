# Movie Recommendation System — Hybrid (Content-Based + Collaborative Filtering)

A movie recommender that combines **content-based filtering** (genres, cast, director, keywords) with **collaborative filtering** (SVD on user ratings) to generate personalized recommendations.

## Why Hybrid?

- **Content-based filtering alone** recommends movies similar in genre/cast/director, but gives the *same* recommendations to every user — it has no concept of individual taste.
- **Collaborative filtering (SVD)** learns from real user rating patterns to personalize predictions, but can't explain *why* two movies are related and struggles with new/unrated items.
- **This hybrid approach** uses content similarity to shortlist relevant candidates, then ranks them per-user using SVD's predicted rating — combining the strengths of both.

## Tech Stack

- **Python**, Pandas, NumPy
- **scikit-learn** — CountVectorizer, cosine similarity
- **scikit-surprise** — SVD-based collaborative filtering
- **NLTK** — stemming for keyword normalization

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) by Rounak Banik (Kaggle) — ~45,000 movies with metadata (genres, cast, crew, keywords) and ~100,000 user ratings.

## Approach

1. **Data Cleaning & Merging** — merged movie metadata, credits, and keywords into a unified dataset.
2. **Content-Based Filtering** — built a metadata "soup" per movie (genres + top 3 cast + director + filtered keywords), vectorized with `CountVectorizer`, and computed pairwise cosine similarity.
3. **Collaborative Filtering** — trained an SVD model on the full user-ratings matrix (~100,000 ratings across 671 users), evaluated with 5-fold cross-validation.
4. **Hybrid Model** — for a given user + movie, shortlist the top 25 content-similar movies, then rank them by the user's SVD-predicted rating.
5. **Evaluation** — held out 20% of all ratings (across all users) as a test set; measured RMSE (rating prediction accuracy) and Precision@10 (top-K recommendation quality) on that held-out set.

## Results

| Metric | Score |
|---|---|
| RMSE (held-out test set) | ~0.90 |
| Precision@10 | ~0.72 |

These metrics reflect overall model performance across the full dataset — not a single user. The hybrid model also produces genuinely personalized rankings: the same movie query returns different top recommendations for different users, based on their individual rating history.

## Limitations

- **Cold start**: new users or movies with no ratings can't be personalized via the collaborative component.
- **Static dataset**: snapshot from ~2017, doesn't include recent releases.
- **Precision@10** could likely improve with feature weighting or a larger ratings dataset.

## How to Run

1. Open `movie-recommendation-system.ipynb` in Google Colab or Kaggle Notebooks.
2. Run all cells top to bottom (dataset auto-downloads via `kagglehub`).
3. Use the interactive cells to enter a movie title and user ID to get personalized recommendations.

## Author

Kushagra Shrikhande
