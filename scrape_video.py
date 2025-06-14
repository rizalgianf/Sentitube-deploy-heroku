from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from dotenv import load_dotenv
import fasttext
from tensorflow.keras.models import load_model
import numpy as np
import os
import re
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Load FastText dan LSTM model
FT_MODEL_PATH = "fasttext_model.bin"      # ganti sesuai nama file fasttext-mu
LSTM_MODEL_PATH = "lstm_model.h5"         # ganti sesuai nama file lstm-mu
ft_model = fasttext.load_model(FT_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

LABELS = ["negatif", "netral", "positif"]  # urutkan sesuai label encoder training

app = Flask(__name__)
CORS(app)  # Izinkan akses dari semua origin

def extract_video_id(url):
    """
    Ekstrak video ID dari berbagai format link YouTube.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", " ", text)  # Hapus link, mention, karakter selain huruf/spasi
    text = re.sub(r"[^\x00-\x7F]+", "", text)           # Hapus emoji & karakter non-ASCII
    text = text.replace("\n", " ")                      # Hapus newline
    return text.strip()

def get_replies(youtube, parent_id, video_id):
    replies = []
    next_page_token = None
    while True:
        reply_request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        )
        reply_response = reply_request.execute()
        for item in reply_response.get('items', []):
            comment = item['snippet']
            replies.append({
                'Timestamp': comment['publishedAt'],
                'Username': comment['authorDisplayName'],
                'VideoID': video_id,
                'Comment': comment['textDisplay'],
                'Cleaned': clean_text(comment['textDisplay']),
                'Date': comment.get('updatedAt', comment['publishedAt'])
            })
        next_page_token = reply_response.get('nextPageToken')
        if not next_page_token:
            break
    return replies

def get_comments_for_video(youtube, video_id):
    all_comments = []
    next_page_token = None
    while True:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            textFormat="plainText",
            maxResults=100
        )
        comment_response = comment_request.execute()
        for item in comment_response.get('items', []):
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Timestamp': top_comment['publishedAt'],
                'Username': top_comment['authorDisplayName'],
                'VideoID': video_id,
                'Comment': top_comment['textDisplay'],
                'Cleaned': clean_text(top_comment['textDisplay']),
                'Date': top_comment.get('updatedAt', top_comment['publishedAt'])
            })
            if item['snippet']['totalReplyCount'] > 0:
                all_comments.extend(get_replies(youtube, item['snippet']['topLevelComment']['id'], video_id))
        next_page_token = comment_response.get('nextPageToken')
        if not next_page_token:
            break
    # Hapus komentar yang hasil clean-nya kosong
    all_comments = [c for c in all_comments if c['Cleaned'].strip()]
    return all_comments

def get_video_details(youtube, video_id):
    request = youtube.videos().list(
        part="snippet,statistics,status,contentDetails,player",
        id=video_id
    )
    response = request.execute()
    if not response['items']:
        return None
    info = response['items'][0]
    return {
        'video_id': video_id,
        'title': info['snippet']['title'],
        'description': info['snippet']['description'],
        'channel_title': info['snippet']['channelTitle'],
        'channel_id': info['snippet']['channelId'],
        'published_at': info['snippet']['publishedAt'],
        'view_count': info['statistics'].get('viewCount'),
        'like_count': info['statistics'].get('likeCount'),
        'comment_count': info['statistics'].get('commentCount'),
        'tags': info['snippet'].get('tags', []),
        'category_id': info['snippet'].get('categoryId'),
        'privacy_status': info['status']['privacyStatus'],
        'duration': info.get('contentDetails', {}).get('duration'),
        'definition': info.get('contentDetails', {}).get('definition'),
        'caption': info.get('contentDetails', {}).get('caption'),
        'thumbnails': info['snippet'].get('thumbnails', {}),
        'embed_html': info.get('player', {}).get('embedHtml')
    }

@app.route('/scrape_comments', methods=['POST'])
def scrape_comments():
    data = request.json
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'error': 'video_url is required'}), 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube video URL'}), 400

    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        video_details = get_video_details(youtube, video_id)
        all_comments = get_comments_for_video(youtube, video_id)

        if not all_comments:
            return jsonify({'error': 'No comments found'}), 404

        # Prediksi sentimen
        cleaned_texts = [c['Cleaned'] for c in all_comments]
        sentiments = predict_sentiment_lstm(cleaned_texts)
        for c, s in zip(all_comments, sentiments):
            c['Sentiment'] = s

        return jsonify({
            'video_details': video_details,
            'comments': all_comments
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_videos', methods=['POST'])
def search_videos():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'query is required'}), 400
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=10
    ).execute()
    results = []
    for item in search_response.get('items', []):
        results.append({
            'video_id': item['id']['videoId'],
            'title': item['snippet']['title'],
            'channel_title': item['snippet']['channelTitle'],
            'published_at': item['snippet']['publishedAt'],
            'thumbnail': item['snippet']['thumbnails']['high']['url']
        })
    return jsonify(results)

def predict_sentiment_lstm(texts):
    vectors = np.vstack([ft_model.get_sentence_vector(t) for t in texts])
    vectors = vectors.reshape(vectors.shape[0], vectors.shape[1], 1)
    preds = lstm_model.predict(vectors)
    labels = [LABELS[np.argmax(p)] for p in preds]
    return labels

if __name__ == '__main__':
    app.run(debug=True)