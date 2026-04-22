🧠 Project Architecture
The system follows a Two-Stage Pipeline approach:

Stage 1: Candidate Generation (SVD) * Uses a PyTorch-based SVD (Singular Value Decomposition) model.

Learns user and item embeddings to predict ratings for unseen products.

Retrieves the top 20 "mathematical" candidates for a specific User ID.

Stage 2: Sentiment Filtering (1D CNN)

A 1D Convolutional Neural Network (CNN) analyzes the text of the most recent reviews for the top 20 candidates.

If the CNN detects trending negative sentiment, the item is penalized and removed from the list.

Ensures that "safe" and high-quality recommendations reach the user.

🛠️ Tech Stack
Deep Learning Framework: PyTorch (Custom nn.Module architectures)

Language: Python 3.12

Hardware Acceleration: NVIDIA CUDA (Optimized for GTX 1650)

Database: MongoDB (API request logging & data persistence)

Backend: Flask (RESTful API)

Frontend: HTML5

Data Processing: Pandas, NumPy, Scikit-learn

📈 Model Performance
Collaborative Filtering (SVD): Trained with MSE Loss, achieving high convergence on user-item interaction matrices.

Sentiment Classifier (CNN): Achieved ~92.8% Validation Accuracy using a 1D-CNN architecture with Global Max Pooling.

📂 Project Structure
Plaintext
hybrid_recommender/
├── data/               # Raw and Processed CSV datasets
├── models/             # Saved .pth weights and vocab.json
├── notebooks/          # Exploratory Data Analysis & Training experiments
├── src/                
│   ├── recommender.py  # SVD Inference Engine
│   ├── sentiment.py    # CNN Inference Engine
│   └── hybrid_engine.py # Master Pipeline Logic
├── templates/          # HTML Frontend
├── app.py              # Flask Web Server & MongoDB Integration
└── README.md
🚀 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/yourusername/hybrid_recommender.git
cd hybrid_recommender
Install dependencies:

Bash
pip install torch pandas flask pymongo scikit-learn matplotlib
Database Setup:

Ensure MongoDB is running locally on port 27017.

Run the Application:

Bash
python app.py
The browser will automatically open the UI at http://127.0.0.1:5000.

📊 Key Insights for Recruiters
Cold Start & Trending Awareness: Unlike basic SVD models, this system handles trending product quality drops by utilizing real-time sentiment analysis.

Systemic Reliability: Includes a MongoDB logging layer to track API performance and audit AI decisions.

Optimized for Edge: Developed to run efficiently on consumer-grade GPUs (NVIDIA GTX 1650) using custom PyTorch optimization.

