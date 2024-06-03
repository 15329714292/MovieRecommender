import csv

# 给定的 genre2id 字典
genre2id = {'Action': 0, 'Adventure': 1, 'Animation': 2, 
            "Children's": 3, 'Comedy': 4, 'Crime': 5, 
            'Documentary': 6, 'Drama': 7, 'Fantasy': 8, 
            'Film-Noir': 9, 'Horror': 10, 'Musical': 11, 
            'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 
            'Thriller': 15, 'War': 16, 'Western': 17}

# 读取 movies.csv 并解析 genres
movies = []
with open('movies.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过头行
    for row in reader:
        movie_id = int(row[0])
        genres = row[2].split('/')
        genre_ids = [genre2id[genre] for genre in genres if genre in genre2id]
        movies.append((movie_id, genre_ids))

# 保存 movie_id 对应的 genre_ids 到 movie2genres.csv
with open('movie2genres.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['MOVIE_ID', 'GENRE_IDS'])
    for movie_id, genre_ids in movies:
        writer.writerow([movie_id, ','.join(map(str, genre_ids))])

print("Preprocessing complete and saved to movie2genres.csv")