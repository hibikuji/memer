import os
import shutil
import json
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import difflib
import speech_recognition as sr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ⚠️ ここにステップ2で取得したDiscordのWebhook_URLを貼り付ける
# ==========================================
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1479128248838389910/NC3lk2f60F0B_UbTK9BNIDUd6Z_u_Ynmx4gyRg9YjqHzWHLeLNwMYLTMNOMKuu2jvCyU"
# ==========================================

# フォルダの設定（重いdatasetフォルダは廃止！）
FEATURES_DIR = "features" # 特徴量（数値データ）だけを保存する軽いフォルダ
AUDIOS_DIR = "audios"
IMAGES_DIR = "images"
MEMES_INFO_FILE = "memes_info.json"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(AUDIOS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

if not os.path.exists(MEMES_INFO_FILE):
    with open(MEMES_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

def load_memes_info():
    with open(MEMES_INFO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meme_info(meme_id, text):
    info = load_memes_info()
    info[meme_id] = {"text": text}
    with open(MEMES_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

def augment_audio(y, sr, num_augments=20):
    augmented_signals = [y]
    for _ in range(num_augments - 1):
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y_noise = y + noise_amp * np.random.normal(size=y.shape[0])
        pitch_steps = np.random.uniform(-2, 2)
        y_pitch = librosa.effects.pitch_shift(y_noise, sr=sr, n_steps=pitch_steps)
        augmented_signals.append(y_pitch)
    return augmented_signals

# ★Web画面を配信する機能を追加！★
@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# AIの学習（WAVではなく、軽い数値データから学習する）
def train_model():
    X = []
    y_labels = []
    for file_name in os.listdir(FEATURES_DIR):
        if file_name.endswith(".pkl"):
            meme_id = file_name.replace(".pkl", "")
            meme_features = joblib.load(os.path.join(FEATURES_DIR, file_name))
            for feat in meme_features:
                X.append(feat)
                y_labels.append(meme_id)
    
    if len(set(y_labels)) < 2:
        return False # 2種類以上のデータが必要
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_scaled, y_labels)
    
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return True

@app.post("/register_meme/")
async def register_meme(meme_id: str=Form(...), text: str=Form(...), audio: UploadFile=File(...), image: UploadFile=File(...)):
    try:
        # 1. 画像の保存
        image_ext = os.path.splitext(image.filename)[1] or ".png"
        shutil.copyfileobj(image.file, open(os.path.join(IMAGES_DIR, f"{meme_id}{image_ext}"), "wb"))

        # 2. 音声を一時的に受け取る
        temp_audio_path = os.path.join(AUDIOS_DIR, f"temp_{audio.filename}")
        shutil.copyfileobj(audio.file, open(temp_audio_path, "wb"))

        # 3. 音声を読み込んでメモリ上で水増し＆特徴抽出（WAVは作らない！）
        y, sr_rate = librosa.load(temp_audio_path, sr=None)
        augmented_signals = augment_audio(y, sr_rate, num_augments=20)
        
        meme_features = []
        for aug_y in augmented_signals:
            mfccs = librosa.feature.mfcc(y=aug_y, sr=sr_rate, n_mfcc=40)
            meme_features.append(np.mean(mfccs.T, axis=0))

        # 4. 数値データだけを保存し、元の音声は即座に削除！
        joblib.dump(meme_features, os.path.join(FEATURES_DIR, f"{meme_id}.pkl"))
        os.remove(temp_audio_path)

        save_meme_info(meme_id, text)
        train_model() 

        return JSONResponse(content={"status": "success", "message": f"{meme_id}を登録し、不要な音声を削除しました！"})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/trigger_meme/")
async def trigger_meme(audio: UploadFile = File(...)):
    try:
        temp_path = os.path.join(AUDIOS_DIR, f"trigger_{audio.filename}")
        shutil.copyfileobj(audio.file, open(temp_path, "wb"))

        recognizer = sr.Recognizer()
        recognized_text = ""
        try:
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data, language="ja-JP")
        except:
            pass

        if not os.path.exists(MODEL_FILE):
            os.remove(temp_path)
            return JSONResponse(content={"status": "error", "message": "登録されたミームがありません。"})
        
        svm = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        y, sr_rate = librosa.load(temp_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        
        probabilities = svm.predict_proba(scaler.transform([features]))[0]
        max_prob_index = np.argmax(probabilities)
        predicted_meme_id = svm.classes_[max_prob_index]
        audio_confidence = probabilities[max_prob_index]

        memes_info = load_memes_info()
        target_text = memes_info.get(predicted_meme_id, {}).get("text", "")
        
        if target_text and recognized_text:
            text_similarity = difflib.SequenceMatcher(None, recognized_text, target_text).ratio()
        else:
            text_similarity = 0.0

        is_success = (audio_confidence >= 0.7) and (text_similarity >= 0.7)

        if is_success:
            image_path = None
            for ext in [".png", ".jpg", ".jpeg", ".gif"]:
                p = os.path.join(IMAGES_DIR, f"{predicted_meme_id}{ext}")
                if os.path.exists(p):
                    image_path = p
                    break
            if image_path:
                with open(image_path, "rb") as f:
                    requests.post(DISCORD_WEBHOOK_URL, files={"file": f})

        os.remove(temp_path)

        return JSONResponse(content={
            "status": "success",
            "recognized_text": recognized_text,
            "target_text": target_text,
            "audio_confidence": round(audio_confidence * 100, 1),
            "text_similarity": round(text_similarity * 100, 1),
            "triggered": is_success
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
