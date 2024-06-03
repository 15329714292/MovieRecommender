from numpy import shape
import torch
import torch.nn as nn


# Matrix Factorization
class MF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        
    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        output = (user_emb * movie_emb).sum(dim=1)
        return output


# Alternating Least Squares
class ALS(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, lambda_reg=1):
        super(ALS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Initialize embeddings with small random values
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        ratings = (user_embeds * item_embeds).sum(dim=1)
        return ratings

    def user_step(self, user_ratings, item_embedding):
        num_items = item_embedding.size(0)
        lambda_identity = self.lambda_reg * torch.eye(self.embedding_dim, device=item_embedding.device)
        for user_id in range(self.num_users):
            # Extract ratings for the current user
            user_rated_items = user_ratings[user_id].nonzero(as_tuple=True)[0]
            if len(user_rated_items) == 0:
                continue
            # Gather item embeddings and ratings
            item_embeds = item_embedding[user_rated_items]
            ratings = user_ratings[user_id, user_rated_items].float()
            # Compute ALS update for user embeddings
            A = item_embeds.t() @ item_embeds + lambda_identity
            V = item_embeds.t() @ ratings
            self.user_embedding.weight.data[user_id] = torch.linalg.solve(A, V)

    def item_step(self, item_ratings, user_embedding):
        num_users = user_embedding.size(0)
        lambda_identity = self.lambda_reg * torch.eye(self.embedding_dim, device=user_embedding.device)
        for item_id in range(self.num_items):
            # Extract ratings for the current item
            item_rated_users = item_ratings[:, item_id].nonzero(as_tuple=True)[0]
            if len(item_rated_users) == 0:
                continue
            # Gather user embeddings and ratings
            user_embeds = user_embedding[item_rated_users]
            ratings = item_ratings[item_rated_users, item_id].float()
            # Compute ALS update for item embeddings
            A = user_embeds.t() @ user_embeds + lambda_identity
            V = user_embeds.t() @ ratings
            self.item_embedding.weight.data[item_id] = torch.linalg.solve(A, V)


# Generalized Matrix Factorization
class GMF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, use_NeuMF=False):
        super(GMF, self).__init__()
        self.use_NeuMF = use_NeuMF
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        if not use_NeuMF:
            self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        output = user_embeds * movie_embeds
        if not self.use_NeuMF:
            output = self.output_layer(output)
            return output.squeeze()
        return output


# Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, use_NeuMF=False):
        super(MLP, self).__init__()
        self.use_NeuMF = use_NeuMF
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        self.mlp_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.mlp_layer(concat_embeds)
        if not self.use_NeuMF:
            output = self.output_layer(output)
            return output.squeeze()
        return output


# Neural Matrix Factorization
class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, gmf_embedding_dim=32, mlp_embedding_dim=32):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_movies, gmf_embedding_dim, use_NeuMF=True)
        self.mlp = MLP(num_users, num_movies, mlp_embedding_dim, use_NeuMF=True)
        
        self.output_layer = nn.Linear(gmf_embedding_dim + mlp_embedding_dim, 1)

    def forward(self, user_ids, movie_ids):
        gmf_output = self.gmf(user_ids, movie_ids)
        mlp_output = self.mlp(user_ids, movie_ids)
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output_layer(concat_output)
        return output.squeeze()


# Wide & deep
class WideDeep(nn.Module):
    def __init__(
        self, 
        num_users, 
        num_items, 
        user_dim=32, 
        item_dim=32, 
        item_feat_dim=0, 
        dnn_hidden_units=[64, 32], 
        dnn_dropout=0.2, 
        dnn_batch_norm=True
    ):
        super(WideDeep, self).__init__()
        
        # Wide part: Memorization
        self.user_embedding = nn.Embedding(num_users, 1)
        self.item_embedding = nn.Embedding(num_items, 1)
        self.cross_embedding = nn.Embedding(num_users * num_items, 1) # Feature Crosses
        
        # Deep part: Generalization
        self.user_embed = nn.Embedding(num_users, user_dim)
        self.item_embed = nn.Embedding(num_items, item_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.cross_embedding.weight, std=0.01)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

        if item_feat_dim > 0:
            self.item_feat_layer = nn.Linear(item_feat_dim, item_feat_dim)
            dnn_input_dim = user_dim + item_dim + item_feat_dim
        else:
            dnn_input_dim = user_dim + item_dim

        layers = []
        for hidden_unit in dnn_hidden_units:
            layers.append(nn.Linear(dnn_input_dim, hidden_unit))
            if dnn_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit))
            layers.append(nn.ReLU())
            if dnn_dropout > 0:
                layers.append(nn.Dropout(dnn_dropout))
            dnn_input_dim = hidden_unit

        self.deep_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dnn_input_dim, 1)

    def forward(self, user_ids, item_ids, item_feats=None):
        # Wide part
        user_wide = self.user_embedding(user_ids)
        item_wide = self.item_embedding(item_ids)
        cross_wide = self.cross_embedding(user_ids * item_ids)
        wide_output = user_wide + item_wide + cross_wide

        # Deep part
        user_deep = self.user_embed(user_ids)
        item_deep = self.item_embed(item_ids)
        if item_feats is not None:
            item_feats = self.item_feat_layer(item_feats)
            deep_input = torch.cat([user_deep, item_deep, item_feats], dim=-1)
        else:
            deep_input = torch.cat([user_deep, item_deep], dim=-1)

        deep_output = self.deep_layers(deep_input)
        deep_output = self.output_layer(deep_output)
        
        # Combining both parts
        final_output = wide_output + deep_output
        return final_output.squeeze()
    

# Self-Made
class NCF_plus(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=32):
        super(NCF_plus, self).__init__()
        self.user_memory = nn.Embedding(num_users, 1)
        self.movie_memory = nn.Embedding(num_movies, 1)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim + embedding_dim//2)
        self.movie_embedding_gmf = nn.Embedding(num_movies, embedding_dim)
        self.genres_embedding_gmf = nn.EmbeddingBag(num_genres, embedding_dim//2, mode='mean')
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim + embedding_dim//2)
        self.movie_embedding_mlp = nn.Embedding(num_movies, embedding_dim)
        self.genres_embedding_mlp = nn.EmbeddingBag(num_genres, embedding_dim//2, mode='mean')
        self._init_weights()

        self.mlp_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim + embedding_dim//2),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(embedding_dim * 3, 1)

    def _init_weights(self):
        nn.init.normal_(self.user_memory.weight, std=0.01)
        nn.init.normal_(self.movie_memory.weight, std=0.01)
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.movie_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.genres_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.movie_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.genres_embedding_mlp.weight, std=0.01)

    def forward(self, user_ids, movie_ids, genre_ids, offsets):
        user_mem = self.user_memory(user_ids)
        movie_mem = self.movie_memory(movie_ids)
        
        user_embed_gmf = self.user_embedding_gmf(user_ids)
        movie_embed_gmf = torch.cat([self.movie_embedding_gmf(movie_ids), 
                                     self.genres_embedding_gmf(genre_ids, offsets)], dim=1)
        gmf_output = user_embed_gmf * movie_embed_gmf

        user_embed_mlp = self.user_embedding_mlp(user_ids)
        movie_embed_mlp = torch.cat([self.movie_embedding_mlp(movie_ids), 
                                     self.genres_embedding_mlp(genre_ids, offsets)], dim=1)

        mlp_output = self.mlp_layer(torch.cat([user_embed_mlp, movie_embed_mlp], dim=1))

        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output_layer(concat_output)
        
        output = user_mem + movie_mem + output
        return output.squeeze()