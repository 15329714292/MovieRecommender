import torch
from models import MF, ALS, GMF, MLP, NeuMF, WideDeep, NCF_plus


def build_model(use_model, num_users, num_movies, device, num_genres):
    assert use_model in ['MF', 'ALS', 'GMF', 'MLP', 'NeuMF', 'WideDeep', 'NCF_plus']
    if use_model == 'MF':
        model = MF(num_users+1, num_movies+1).to(device)
    elif use_model == 'ALS':
        model = ALS(num_users+1, num_movies+1).to(device)
    elif use_model == 'GMF':
        model = GMF(num_users+1, num_movies+1).to(device)
    elif use_model == 'MLP':
        model = MLP(num_users+1, num_movies+1).to(device)
    elif use_model == 'NeuMF':
        model = NeuMF(num_users+1, num_movies+1).to(device)
    elif use_model == 'WideDeep':
        model = WideDeep(num_users+1, num_movies+1).to(device)
    elif use_model == 'NCF_plus':
        model = NCF_plus(num_users+1, num_movies+1, num_genres).to(device)
    return model


def get_genres(use_dataset):
    if use_dataset == 'movielens':
        genre2id = {'Action': 0, 'Adventure': 1, 'Animation': 2, 
                    'Children\'s': 3, 'Comedy': 4, 'Crime': 5, 
                    'Documentary': 6, 'Drama': 7, 'Fantasy': 8, 
                    'Film-Noir': 9, 'Horror': 10, 'Musical': 11, 
                    'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 
                    'Thriller': 15, 'War': 16, 'Western': 17}
        num_genres = len(genre2id)
    elif use_dataset == 'douban':
        genre2id = {'爱情': 0, '动画': 1, '动作': 2, '短片': 3, '儿童': 4, 
                    '犯罪': 5, '歌舞': 6, '鬼怪': 7, '古装': 8, '荒诞': 9, 
                    '家庭': 10, '纪录片': 11, '惊悚': 12, '剧情': 13, 
                    '科幻': 14, '恐怖': 15, '历史': 16, '冒险': 17, 
                    '奇幻': 18, '情色': 19, '同性': 20, '脱口秀': 21, 
                    '舞台艺术': 22, '武侠': 23, '西部': 24, '喜剧': 25, 
                    '戏曲': 26, '悬疑': 27, '音乐': 28, '运动': 29, 
                    '灾难': 30, '战争': 31, '真人秀': 32, '传记': 33}
        num_genres = len(genre2id)
    return genre2id, num_genres


def get_genres_input(movies, movieid2genreids, device):
    genre_ids = []
    offsets = [0]
    
    for movie_id in movies.cpu().numpy():
        genre_ids.extend(movieid2genreids[movie_id])
        offsets.append(len(genre_ids))
    
    genre_ids_tensor = torch.tensor(genre_ids, dtype=torch.long, device=device)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
    
    return genre_ids_tensor, offsets_tensor
