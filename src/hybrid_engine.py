import pandas as pd
from recommender import RecommenderEngine
from sentiment import SentimentEngine

class HybridEngine:
    def __init__(self):
        print("Initializing Master Hybrid Pipeline...")
        # 1. Boot up both AI Brains
        self.ml_engine = RecommenderEngine()
        self.dl_engine = SentimentEngine()
        
        # 2. Load the dataset to look up real reviews for the recommended items
        self.data_path = r"C:\Users\Tejas\Desktop\hybrid_recommender\data\processed\cleaned_reviews.csv"
        self.df = pd.read_csv(self.data_path)
        print("Master Hybrid Pipeline Ready!")

    def get_hybrid_recommendations(self, user_id, initial_candidates=20, final_top_k=5):
        print(f"\n--- Generating Hybrid Recommendations for User {user_id} ---")
        
        # Phase 1: The Broad Net (Mathematical SVD)
        candidates = self.ml_engine.get_top_items(user_id, top_n=initial_candidates)
        print(f"Phase 1: SVD generated {len(candidates)} mathematical candidates.")
        
        verified_recommendations = []
        
        # Phase 2: The Context Check (CNN Sentiment)
        for item_id in candidates:
            # Fetch up to 3 of the most recent text reviews for this specific item
            item_reviews = self.df[self.df['item_id_int'] == item_id]['clean_text'].dropna().head(3).tolist()
            
            # If an item has no text reviews to analyze, we skip it to be safe
            if not item_reviews:
                continue 
                
            positive_votes = 0
            # Run each review through the Deep Learning model
            for review in item_reviews:
                label, conf = self.dl_engine.predict_sentiment(review)
                if label == 1:
                    positive_votes += 1
                    
            # Phase 3: The Filter
            # The item must have a majority positive sentiment from its recent reviews
            if positive_votes >= (len(item_reviews) / 2):
                verified_recommendations.append(item_id)
            else:
                print(f"  -> PENALTY: Filtered out Item {item_id} due to trending negative reviews.")
                
            # Stop computing once we hit our target number of safe recommendations
            if len(verified_recommendations) == final_top_k:
                break
                
        print(f"Phase 4: CNN verified {len(verified_recommendations)} context-safe items.")
        return verified_recommendations

# Test the completed pipeline
if __name__ == "__main__":
    engine = HybridEngine()
    
    test_user = 0
    # Ask SVD for 20 ideas, use CNN to filter it down to the best 5
    final_recs = engine.get_hybrid_recommendations(user_id=test_user, initial_candidates=20, final_top_k=5)
    
    print(f"\nFINAL OUTPUT - Top {len(final_recs)} Recommendations for User {test_user}:")
    print(final_recs)