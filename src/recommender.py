import torch
import torch.nn as nn
import pandas as pd
import os

# 1. The Exact Blueprint from Day 2
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        u_bias = self.user_bias(user_ids).squeeze()
        i_bias = self.item_bias(item_ids).squeeze()
        dot_product = (u * i).sum(dim=1)
        return dot_product + u_bias + i_bias + self.global_bias

# 2. The Inference Engine
class RecommenderEngine:
    def __init__(self):
        print("Initializing ML Recommender Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.data_path = r"C:\Users\Tejas\Desktop\hybrid_recommender\data\processed\cleaned_reviews.csv"
        self.model_path = r"C:\Users\Tejas\Desktop\hybrid_recommender\models\svd_model.pth"
        
        # Load dataset to get exact matrix sizes and mappings
        self.df = pd.read_csv(self.data_path)
        self.num_users = self.df['user_id_int'].nunique()
        self.num_items = self.df['item_id_int'].nunique()
        
        # Instantiate and load the model
        self.model = MatrixFactorization(self.num_users, self.num_items, embedding_dim=32)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # CRITICAL: Sets model to inference mode (no learning)
        print("ML Engine Ready!")

    def get_top_items(self, user_id_int, top_n=20):
        """Predicts scores for all items for a given user and returns the top N."""
        
        # Create a tensor of all possible item IDs (0 to num_items - 1)
        all_item_ids = torch.arange(self.num_items).to(self.device)
        
        # Create a tensor of the user ID, repeated to match the number of items
        user_tensor = torch.full((self.num_items,), user_id_int, dtype=torch.long).to(self.device)
        
        # Run inference (turn off gradients to save memory and speed up calculation)
        with torch.no_grad():
            predicted_scores = self.model(user_tensor, all_item_ids)
            
        # Get the indices (item IDs) of the highest scoring items
        top_scores, top_indices = torch.topk(predicted_scores, top_n)
        
        # Return as a clean Python list
        return top_indices.cpu().numpy().tolist()

# 3. Simple test to ensure it works when we run this script directly
if __name__ == "__main__":
    engine = RecommenderEngine()
    
    # Let's test it by asking for recommendations for User ID 0
    test_user = 0
    top_20 = engine.get_top_items(user_id_int=test_user, top_n=20)
    
    print(f"\nTop 20 mathematically recommended Item IDs for User {test_user}:")
    print(top_20)