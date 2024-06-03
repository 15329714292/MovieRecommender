# Movie Recommender

We have explored some commonly used deep learning methods for collaborative filtering recommendations. In addition to the second-order interactions using the dot product of user and movie embeddings in the matrix factorization (MF) method, we incorporated nonlinear interactions with a multi-layer perceptron (MLP). We also added feature embeddings based on movie IDs. We conducted some tests on the Movielens dataset. As a supplement to the model, we implemented some content-based recommendation methods, such as genre preference matching. Specifically, this was implemented by calculating a genre interest vector based on the user's existing rating data, and then performing a dot product operation with the genre TF-IDF vector of each movie to obtain the interest matching score.

To build a usable movie recommendation system, we collected Douban rating data and performed a series of data cleaning tasks. We then trained models based on this dataset. We implemented a simple GUI interfaced with the model, allowing users to rate some movies and then generate a personalized recommendation report.

Run the following code for a quick look:

```
python GUI.py
```
