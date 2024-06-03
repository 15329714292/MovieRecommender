import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import RatingsDataset
from utils import build_model, get_genres, get_genres_input

# Random guess: 3.5979
# Best loss: 0.8258 0.8236 0.7858 0.7995 0.7988 0.8210 0.7577
# Set hyperparameters
use_model = 'NCF_plus'
assert use_model in ['MF', 'ALS', 'GMF', 'MLP', 'NeuMF', 'WideDeep', 'NCF_plus']
use_dataset = 'douban'
assert use_dataset in ['movielens', 'douban']

load_ready_made = True
load_pretrained = False
record_loss = False
early_stop = False
if record_loss:
    # Setup TensorBoard
    writer = SummaryWriter(log_dir='./logs')

epochs = 5

print(f"Model: {use_model}")
print(f"Dataset: {use_dataset}")

# Load data
ratings = pd.read_csv('./data/'+use_dataset+'/ratings.csv')
movies = pd.read_csv('./data/'+use_dataset+'/movies.csv')
num_users = ratings['USER_ID'].max()
num_movies = ratings['MOVIE_ID'].max()
print(f'Number of users: {num_users}')
print(f'Number of movies: {num_movies}')

genre2id, num_genres = get_genres(use_dataset)
if use_model == 'NCF_plus':
    print(f'Number of genres: {num_genres}')
movies['GENRES'] = movies['GENRES'].apply(lambda x: x.split('/'))
movies['GENRES'] = movies['GENRES'].apply(lambda genres: [genre2id[genre] for genre in genres])

movieid2genreids = dict(zip(movies['MOVIE_ID'], movies['GENRES']))

if load_ready_made:
    train_data = pd.read_csv('./data/'+use_dataset+'/ratings_train.csv')
    test_data = pd.read_csv('./data/'+use_dataset+'/ratings_test.csv')
    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')
else:
    # Split the data
    train_data = []
    test_data = []
    print('Splitting data...')
    for user_id, group in tqdm(ratings.groupby('USER_ID')):
        # test_sample = group.sample(n=1, random_state=42)
        test_sample = group.sample(frac=0.2, random_state=42)
        train_samples = group.drop(test_sample.index)
        
        test_data.append(test_sample)
        train_data.append(train_samples)
    # Concatenate the list of DataFrames back into single DataFrames
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    train_data.to_csv('./data/'+use_dataset+'/ratings_train.csv')
    test_data.to_csv('./data/'+use_dataset+'/ratings_test.csv')

    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(use_model, num_users, num_movies, device, num_genres)
if load_pretrained == True:
    model.load_state_dict(torch.load('./weights/'+use_dataset+'/'+use_model+'.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
train_dataset = RatingsDataset(train_data)
test_dataset = RatingsDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
best_test_loss = 1e2

try:
    print('Start training...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for user_ids, movie_ids, ratings in tqdm(train_loader):
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            if not use_model == 'NCF_plus':
                outputs = model(user_ids, movie_ids)
            else:
                genre_ids, offset = get_genres_input(movie_ids, movieid2genreids, device)
                outputs = model(user_ids, movie_ids, genre_ids, offset)
            optimizer.zero_grad()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * user_ids.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        if record_loss:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in test_loader:
                user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
                if not use_model == 'NCF_plus':
                    outputs = model(user_ids, movie_ids)
                else:
                    genre_ids, offset = get_genres_input(movie_ids, movieid2genreids, device)
                    outputs = model(user_ids, movie_ids, genre_ids, offset)
                loss = criterion(outputs, ratings)
                test_loss += loss.item() * user_ids.size(0)

        test_loss /= len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}')
        if record_loss:
            writer.add_scalar('Loss/test', test_loss, epoch)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), './weights/'+use_dataset+'/'+use_model+'.pth')
        elif early_stop:
            break
except KeyboardInterrupt:
    print(f'Training interrupted. Best Test Loss: {best_test_loss:.4f}')
    exit(0)
print(f'Best Test Loss: {best_test_loss:.4f}')
if record_loss:
    writer.close()
