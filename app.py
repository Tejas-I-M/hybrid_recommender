from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import sys
import os
import webbrowser
from threading import Timer
from datetime import datetime

# Path Fix
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from hybrid_engine import HybridEngine

app = Flask(__name__)

# 1. Database Setup
print("Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
db = client["recommender_db"]
logs_collection = db["api_logs"]

# 2. AI Engine Setup
print("Loading AI Brains (SVD + CNN)...")
engine = HybridEngine()

# --- ROUTES ---

@app.route('/')
def home():
    # This serves the beautiful UI we're about to build
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id_str = request.args.get('user_id')
    if not user_id_str:
        return jsonify({"error": "No User ID provided"}), 400
    
    try:
        user_id = int(user_id_str)
        recs = engine.get_hybrid_recommendations(user_id=user_id)
        
        # Log to Mongo
        logs_collection.insert_one({
            "user_id": user_id, 
            "recs": recs, 
            "timestamp": datetime.now()
        })
        
        return jsonify({"user_id": user_id, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. The "Auto-Open" Magic
def open_browser():
    # Opens the browser directly to your recommendation test link
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("\n--- SERVER STARTING ---")
    # Wait 1.5 seconds for the server to boot, then open the browser
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000)