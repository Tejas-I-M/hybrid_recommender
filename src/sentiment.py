import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re

# 1. The Blueprint
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, kernel_size=5):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = F.relu(self.conv1d(embedded))
        pooled = F.max_pool1d(conved, kernel_size=conved.shape[2]).squeeze(2)
        return self.fc(pooled).squeeze(1)

# 2. The Engine
class SentimentEngine:
    def __init__(self):
        print("Initializing DL Sentiment Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = r"C:\Users\Tejas\Desktop\hybrid_recommender\models\sentiment_cnn.pth"
        self.vocab_path = r"C:\Users\Tejas\Desktop\hybrid_recommender\models\vocab.json"
        
        self.MAX_LEN = 160
        
        # Load the EXACT vocabulary used during training
        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        self.vocab_size = len(self.vocab) + 2 
        
        self.model = SentimentCNN(vocab_size=self.vocab_size)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("DL Sentiment Engine Ready!")

    def clean_text(self, text):
        text = str(text).lower()
        return re.sub(r'[^a-z0-9\s]', '', text)

    def predict_sentiment(self, text):
        clean_str = self.clean_text(text)
        
        # json keys are always strings, so we must ensure we look up the word properly
        sequence = [self.vocab.get(word, 1) for word in clean_str.split()]
        if len(sequence) >= self.MAX_LEN:
            sequence = sequence[:self.MAX_LEN]
        else:
            sequence = sequence + [0] * (self.MAX_LEN - len(sequence))
            
        tensor_seq = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            raw_prediction = self.model(tensor_seq)
            confidence = torch.sigmoid(raw_prediction).item()
            
        label = 1 if confidence > 0.5 else 0
        return label, confidence

if __name__ == "__main__":
    engine = SentimentEngine()
    
    test_review_1 = "This product is absolutely amazing, I love it! The best coffee ever."
    test_review_2 = "Terrible experience. It tasted awful and arrived completely destroyed."
    
    lbl1, conf1 = engine.predict_sentiment(test_review_1)
    lbl2, conf2 = engine.predict_sentiment(test_review_2)
    
    print(f"\nReview: '{test_review_1}'\nPrediction: {'Positive' if lbl1==1 else 'Negative'} (Confidence: {conf1:.4f})")
    print(f"\nReview: '{test_review_2}'\nPrediction: {'Positive' if lbl2==1 else 'Negative'} (Confidence: {conf2:.4f})")