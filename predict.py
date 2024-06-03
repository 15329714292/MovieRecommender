import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from PIL import Image
from io import BytesIO
from utils import build_model, get_genres, get_genres_input


# Set hyperparameters
user_id = 0
top_k = 10
output_file = './output/output.md'
use_model = 'NCF_plus'
assert use_model in ['MF', 'ALS', 'GMF', 'MLP', 'NeuMF', 'NCF_plus']
use_dataset = 'douban'
assert use_dataset in ['douban']

# 高分优先 热度优先 新发布优先 兴趣匹配优先
p = [0, 0.05, 0.2, 1]

# Load data
ratings = pd.read_csv('./data/douban/ratings.csv')
movies = pd.read_csv('./data/douban/movies.csv')

num_users = ratings['USER_ID'].max()
num_movies = ratings['MOVIE_ID'].max()
genre2id, num_genres = get_genres(use_dataset)
if use_model == 'NCF_plus':
    movies['GENRES'] = movies['GENRES'].apply(lambda x: x.split('/'))
    movies['GENRES'] = movies['GENRES'].apply(lambda genres: [genre2id[genre] for genre in genres])
    movieid2genreids = dict(zip(movies['MOVIE_ID'], movies['GENRES']))
    movies = pd.read_csv('./data/douban/movies.csv')

# Build model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(use_model, num_users, num_movies, device, num_genres)
model.load_state_dict(torch.load('./weights/douban/'+use_model+'.pth'))

if user_id < 0 or user_id > num_users:
    raise ValueError(f"User ID {user_id} doesn't exist.")

# Fine-tune
if user_id == 0:
    cold_start_ratings = pd.read_csv('./data/douban/cold_start.csv', header=None)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    user_ids = torch.tensor(cold_start_ratings.iloc[:, 0]).to(device)
    movie_ids = torch.tensor(cold_start_ratings.iloc[:, 1]).to(device)
    ratings = torch.tensor(cold_start_ratings.iloc[:, 2]).float().to(device)
    if not use_model == 'NCF_plus':
        outputs = model(user_ids, movie_ids)
    else:
        genre_ids, offset = get_genres_input(movie_ids, movieid2genreids, device)
        outputs = model(user_ids, movie_ids, genre_ids, offset)
    optimizer.zero_grad()
    loss = criterion(outputs, ratings)
    loss.backward()
    optimizer.step()

# Predict ratings
model.eval()
user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
movie_tensor = torch.tensor(range(1, num_movies+1), dtype=torch.long).to(device)
with torch.no_grad():
    if not use_model == 'NCF_plus':
        predicted_ratings = model(user_tensor.repeat(len(movie_tensor)), movie_tensor)
    else:
        genre_ids, offset = get_genres_input(movie_tensor, movieid2genreids, device)
        predicted_ratings = model(user_tensor.repeat(len(movie_tensor)), movie_tensor, genre_ids, offset)
    
predicted_ratings = predicted_ratings.cpu().numpy()

# Compute interest match score
# Index = movie_id - 1
movie_genre_matrix = np.zeros((num_movies, num_genres))
for movie_id, genre_ids in movieid2genreids.items():
    movie_genre_matrix[movie_id - 1, genre_ids] = 1
# Compute the Term Frequency (TF)
tf = movie_genre_matrix / movie_genre_matrix.sum(axis=1, keepdims=True)
# Compute the Inverse Document Frequency (IDF)
df = np.sum(movie_genre_matrix, axis=0)
idf = np.log(num_movies / (df + 1))
tf_idf = tf * idf
if user_id == 0:
    rated_movie_ids = movie_ids.cpu().numpy()
    user_ratings = ratings.cpu().numpy()
else:
    rated_movie_ids = ratings.loc[ratings['USER_ID'] == user_id, 'MOVIE_ID'].values
    user_ratings = ratings.loc[ratings['USER_ID'] == user_id, 'RATING'].values
interest_vector = np.zeros(num_genres)
interest_vector = np.dot(tf_idf[rated_movie_ids - 1].T, user_ratings - 3)
if np.linalg.norm(interest_vector, ord=2) > 0:
    interest_vector /= np.linalg.norm(interest_vector, ord=2)
common_genres = ["剧情", "喜剧", "动作", "爱情", "科幻", "动画", "悬疑", "惊悚", "恐怖"]
for genre in common_genres:
    interest_vector[genre2id[genre]] += 0.1
if np.linalg.norm(interest_vector, ord=2) > 0:
    interest_vector /= np.linalg.norm(interest_vector, ord=2)
# for genre, id in genre2id.items():
#     print(genre, ': ', interest_vector[id])
interest_match_scores = np.dot(tf_idf, interest_vector)

weighted_scores = predicted_ratings + \
                  p[0] * movies['DOUBAN_SCORE'].values + \
                  p[1] * np.log(movies['DOUBAN_VOTES'].values) - \
                  p[2] * np.log(2021 - movies['YEAR'].values) + \
                  p[3] * interest_match_scores
# 去除已读
weighted_scores[rated_movie_ids - 1] = 0

print("Max predict score: ", max(predicted_ratings))
print("Max rating score: ", max(p[0] * movies['DOUBAN_SCORE'].values))
print("Max popularity score: ", max(p[1] * np.log(movies['DOUBAN_VOTES'].values)))
print("Max oldness penalty: ", max(p[2] * np.log(2021 - movies['YEAR'].values)))
print("Max interest score: ", max(p[3] * interest_match_scores))

top_movie_ids = weighted_scores.argsort()[-top_k:][::-1]

with open(output_file, 'w', encoding='utf-8') as f:
    rank = 1
    for id in top_movie_ids:
        if not pd.isnull(movies.loc[id,'COVER']):
            cover_url = movies.loc[id,'COVER']
            cover_path = f"./output/assets/{id}.jpg"
            response = requests.get(cover_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image.save(cover_path)
                f.write(f"![封面](./assets/{id}.jpg)\n")
            else:
                print(f"Failed to download image {id}")

        f.write(f"## {rank}、{movies.loc[id,'NAME']}\n")
        rank += 1
        f.write(f"**年代**：{movies.loc[id,'YEAR']:.0f}\n")
        f.write(f"**产地**：{movies.loc[id,'REGIONS']}\n")
        f.write(f"**语言**：{movies.loc[id,'LANGUAGES']}\n")
        
        if not pd.isnull(movies.loc[id,'GENRES']):
            f.write(f"**类别**：{movies.loc[id,'GENRES']}\n")
        
        if movies.loc[id,'DOUBAN_SCORE'] != 0:
            f.write(f"**豆瓣评分**：{movies.loc[id,'DOUBAN_SCORE']:.1f}/10 ({movies.loc[id,'DOUBAN_VOTES']:.0f}人评价)\n")
        
        if not pd.isnull(movies.loc[id,'DIRECTORS']):
            f.write(f"**导演**：{movies.loc[id,'DIRECTORS']}\n")
        
        if not pd.isnull(movies.loc[id,'ACTORS']):
            f.write(f"**主演**：{movies.loc[id,'ACTORS']}\n")
        
        if not pd.isnull(movies.loc[id,'STORYLINE']):
            f.write(f"###### **简介**：{movies.loc[id,'STORYLINE']}\n")
        
        f.write('-' * 80 + '\n\n')

