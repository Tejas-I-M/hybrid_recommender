🚀 Hybrid AI Recommendation Engine

SVD Matrix Factorization + CNN Sentiment Analysis

An end-to-end recommendation system that combines mathematical user-item collaborative filtering with deep learning-based sentiment analysis.
📂 Project Structure
```text
hybrid_recommender/
├── data/               # Raw and Processed CSV datasets
├── models/             # Saved .pth weights and vocab.json
├── notebooks/          # EDA & Training Jupyter Notebooks
├── src/                # Core Logic Scripts
│   ├── recommender.py  # SVD Inference Engine
│   ├── sentiment.py    # CNN Inference Engine
│   └── hybrid_engine.py# Master Pipeline Logic
├── templates/          # HTML Frontend UI
├── app.py              # Flask Web Server & MongoDB Integration
└── README.md           # Installation & Setup Guide
```

🧠 How It Works

The system follows a Two-Stage Pipeline approach to ensure high-quality recommendations:

Candidate Generation (SVD): A PyTorch-based Matrix Factorization model predicts the top 20 items a user is likely to interact with based on historical data.

Sentiment Filtering (1D CNN): A Deep Learning CNN reads the most recent reviews for those 20 items. If an item is currently trending with negative sentiment, it is filtered out.

🛠️ Tech Stack

Deep Learning: PyTorch (SVD & 1D-CNN)

Database: MongoDB (Logging & Persistence)

Backend: Flask (Python)

Frontend: HTML5, CSS3, JavaScript

Data: Pandas, NumPy, Scikit-learn

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/yourusername/hybrid_recommender.git
cd hybrid_recommender


Setup .gitignore (Crucial for GitHub):
Create a file named .gitignore to avoid pushing large data files:

data/
models/*.pth
__pycache__/


Run the Application:

python app.py



Developed By

Tejas Data Science @ IIT Mandi | Masai School
