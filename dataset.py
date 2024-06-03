import torch
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(self, data):
        self.users = data['USER_ID'].values
        self.movies = data['MOVIE_ID'].values
        self.ratings = data['RATING'].values
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(movie, dtype=torch.long), torch.tensor(rating, dtype=torch.float)