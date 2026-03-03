# importing necessary libraries
import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import cv2
import base64
import json
import time

# Use the directory where the script is located as the base path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
projectroot = os.path.dirname(BASE_PATH)
# Paths relative to the script location (Root of workspace)
SER_MODEL_PATH = os.path.join(projectroot, "models", "ser_model.h5")
SER_CLASSES_PATH = os.path.join(projectroot, "models", "emotion_classes.npy")

# FER Model path (in subdirectory)
FER_MODEL_PATH = os.path.join(projectroot, "models","fer_model.h5")

# Audio Config
SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# FER Classes (Fixed)
FER_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Page Config
st.set_page_config(page_title="Multimodal Emotion Analysis", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads SER and FER models."""
    models = {}
    
    # Load SER
    if os.path.exists(SER_MODEL_PATH):
        try:
            models['ser'] = load_model(SER_MODEL_PATH)
            models['ser_type'] = 'legacy' # Keep legacy preprocessing for my_audio_emotion_model.h5
            models['ser_classes'] = np.load(SER_CLASSES_PATH)
        except Exception as e:
            st.error(f"Failed to load SER model: {e}")
            
    # Load FER
    if os.path.exists(FER_MODEL_PATH):
        try:
            models['fer'] = load_model(FER_MODEL_PATH)
            # Load Haar Cascade
            models['face_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            st.error(f"Failed to load FER model: {e}")
            
    return models

# --- AUDIO PROCESSING FUNCTIONS ---
def extract_features_for_diarization(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    features = np.concatenate([
        mfcc_mean, mfcc_std, 
        [np.mean(cent), np.std(cent), np.mean(bw), np.std(bw), np.mean(rolloff), np.std(rolloff)]
    ])
    return features

def preprocess_for_emotion_advanced(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return np.expand_dims(mel_spec_db.T, axis=-1)

def preprocess_for_emotion_legacy(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfccs.T

def process_audio(file_path, model, classes, model_type, num_speakers=2):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        st.warning(f"Could not extract audio (likely missing ffmpeg): {e}")
        return pd.DataFrame()

    intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
    segments_data = []
    diarization_features = []
    
    target_len = int(DURATION * SAMPLE_RATE)
    stride_samples = int(1.0 * sr)

    for start_idx, end_idx in intervals:
        segment_audio = y[start_idx:end_idx]
        if len(segment_audio) / sr < 0.5: continue
            
        start_time = start_idx / sr
        end_time = end_idx / sr
        
        # Diarization
        d_feat = extract_features_for_diarization(segment_audio, sr)
        diarization_features.append(d_feat)

        # Emotion
        predictions = []
        if len(segment_audio) > target_len:
            for i in range(0, len(segment_audio) - target_len + 1, stride_samples):
                chunk = segment_audio[i : i + target_len]
                inp = preprocess_for_emotion_advanced(chunk, sr) if model_type == 'advanced' else preprocess_for_emotion_legacy(chunk, sr)
                predictions.append(model.predict(np.expand_dims(inp, axis=0), verbose=0)[0])
        
        if not predictions:
            padded = np.pad(segment_audio, (0, max(0, target_len - len(segment_audio))), 'constant')[:target_len]
            inp = preprocess_for_emotion_advanced(padded, sr) if model_type == 'advanced' else preprocess_for_emotion_legacy(padded, sr)
            predictions.append(model.predict(np.expand_dims(inp, axis=0), verbose=0)[0])
            
        avg_pred = np.mean(predictions, axis=0)
        max_idx = np.argmax(avg_pred)
        
        segments_data.append({
            "Start": start_time,
            "End": end_time,
            "Emotion": classes[max_idx],
            "Confidence": float(avg_pred[max_idx]),
            "Duration": end_time - start_time,
            "Type": "Audio"
        })

    if not segments_data: return pd.DataFrame()

    # Diarization Clustering
    if len(diarization_features) >= num_speakers and num_speakers > 1:
        try:
            scaler = StandardScaler()
            X_diar = scaler.fit_transform(np.array(diarization_features))
            kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=20)
            speakers = kmeans.fit_predict(X_diar)
             # Smoothing
            if len(speakers) > 2:
                for j in range(1, len(speakers) - 1):
                    if speakers[j-1] == speakers[j+1] and speakers[j] != speakers[j-1]:
                        speakers[j] = speakers[j-1]
        except:
            speakers = [0] * len(segments_data)
    else:
        speakers = [0] * len(segments_data)

    for i, data in enumerate(segments_data):
        data["Speaker"] = f"Speaker {speakers[i] + 1}"

    return pd.DataFrame(segments_data)

# --- VIDEO PROCESSING FUNCTIONS ---
def process_video_fer(video_path, fer_model, face_cascade, rotation_mode="None"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    fer_data = []
    
    # Process every 0.5 seconds
    step = int(fps * 0.5) if fps > 0 else 15
    
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Apply rotation if requested
        if rotation_mode == "90° Clockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == "180°":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_mode == "90° Counter-Clockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frame_idx % step == 0:
            status_text.text(f"Processing frame {frame_idx}/{total_frames}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Find largest face
                max_area = 0
                best_face = faces[0]
                for (x, y, w, h) in faces:
                    if w*h > max_area:
                        max_area = w*h
                        best_face = (x, y, w, h)
                
                x, y, w, h = best_face
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = roi_gray.astype('float32') / 255.0
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels = np.expand_dims(img_pixels, axis=-1)
                    
                    probs = fer_model.predict(img_pixels, verbose=0)[0]
                    max_idx = np.argmax(probs)
                    emotion = FER_CLASSES[max_idx]
                    confidence = float(probs[max_idx])
                    
                    fer_data.append({
                        "Time": frame_idx / fps,
                        "Emotion": emotion,
                        "Confidence": confidence,
                        "Type": "Video"
                    })
                except Exception as e:
                    pass
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            prog_bar.progress(min(1.0, frame_idx / max(1, total_frames)))
            
    cap.release()
    prog_bar.empty()
    status_text.empty()
    return pd.DataFrame(fer_data)

# --- INTERACTIVE PLAYER HTML ---
def get_multimodal_player_html(video_path, df_audio, df_video):
    """
    Generates an HTML video player with synchronized Emotion/Speaker display and Fusion logic.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode()
    # Detect mime type roughly
    ext = os.path.splitext(video_path)[1].lower().replace('.', '')
    mime = f"video/{ext}" if ext != 'mkv' else 'video/mp4' 

    # Convert DataFrames to JSON for JS (Ensure native types)
    audio_records = []
    if not df_audio.empty:
        # Explicitly cast to float/str to avoid numpy serialization issues
        temp_df = df_audio.copy()
        temp_df['Start'] = temp_df['Start'].astype(float)
        temp_df['End'] = temp_df['End'].astype(float)
        temp_df['Confidence'] = temp_df['Confidence'].astype(float)
        temp_df['Speaker'] = temp_df['Speaker'].astype(str)
        temp_df['Emotion'] = temp_df['Emotion'].astype(str)
        audio_records = temp_df[['Start', 'End', 'Speaker', 'Emotion', 'Confidence']].to_dict(orient='records')

    video_records = []
    if not df_video.empty:
        temp_df = df_video.copy()
        temp_df['Time'] = temp_df['Time'].astype(float)
        temp_df['Confidence'] = temp_df['Confidence'].astype(float)
        temp_df['Emotion'] = temp_df['Emotion'].astype(str)
        video_records = temp_df[['Time', 'Emotion', 'Confidence']].to_dict(orient='records')
    
    json_audio = json.dumps(audio_records)
    json_video = json.dumps(video_records)

    html = f"""
    <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333; font-family: sans-serif; text-align: center;">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px;">
            <div id="speaker-box" style="padding: 10px; border-radius: 6px; background-color: #333; color: #888; text-align: center; font-weight: bold; font-size: 0.9em;">Unknown</div>
            <div id="audio-emo-box" style="padding: 10px; border-radius: 6px; background-color: #333; color: #888; text-align: center; font-weight: bold; font-size: 0.9em;">Audio: ...</div>
            <div id="video-emo-box" style="padding: 10px; border-radius: 6px; background-color: #333; color: #888; text-align: center; font-weight: bold; font-size: 0.9em;">Video: ...</div>
        </div>
        <div id="fusion-box" style="padding: 10px; margin-bottom: 15px; border-radius: 6px; background-color: #333; color: #888; text-align: center; font-weight: bold; font-size: 1.1em; border: 1px solid #555;">Fused: ...</div>

        <div style="width: 100%; display: flex; justify-content: center; background: black; border-radius: 8px; overflow: hidden;">
            <video id="video-player" controls style="width: auto; max-width: 100%; height: auto; max-height: 75vh;">
                <source src="data:{mime};base64,{video_base64}" type="{mime}">
                Your browser does not support the video element.
            </video>
        </div>

        <script>
            const audioSegments = {json_audio};
            const videoFrames = {json_video};
            
            const video = document.getElementById('video-player');
            const speakerBox = document.getElementById('speaker-box');
            const audioBox = document.getElementById('audio-emo-box');
            const videoBox = document.getElementById('video-emo-box');
            const fusionBox = document.getElementById('fusion-box');

            const colors = {{
                'neutral': '#D3D3D3', 'calm': '#ADD8E6', 'happy': '#FFD700',
                'sad': '#4682B4', 'angry': '#FF4500', 'fearful': '#800080',
                'disgust': '#008000', 'surprised': '#FFA500',
                'Neutral': '#D3D3D3', 'Happy': '#FFD700', 'Sad': '#4682B4', 
                'Angry': '#FF4500', 'Fear': '#800080', 'Disgust': '#008000', 
                'Surprise': '#FFA500'
            }};

            function updatePlayer() {{
                const t = video.currentTime;
                
                let currAudio = null;
                let currVideo = null;
                
                // 1. Audio Update
                const activeAudio = audioSegments.find(s => t >= s.Start && t < s.End);
                if (activeAudio) {{
                    currAudio = activeAudio.Emotion.toLowerCase();
                    // Normalize SER labels to match FER for fusion comparison
                    if (currAudio === 'fearful') currAudio = 'fear';
                    if (currAudio === 'surprised') currAudio = 'surprise';
                    if (currAudio === 'calm') currAudio = 'neutral'; // Optional: map calm to neutral

                    speakerBox.innerText = activeAudio.Speaker;                    speakerBox.style.backgroundColor = '#444'; 
                    speakerBox.style.color = 'white';

                    audioBox.innerText = "Audio: " + activeAudio.Emotion.toUpperCase();
                    audioBox.style.backgroundColor = colors[currAudio] || '#333';
                    audioBox.style.color = '#000';
                }} else {{
                    speakerBox.innerText = "Silence";
                    speakerBox.style.backgroundColor = '#222';
                    speakerBox.style.color = '#555';

                    audioBox.innerText = "Audio: ...";
                    audioBox.style.backgroundColor = '#222';
                    audioBox.style.color = '#555';
                }}
                
                // 2. Video Update
                // Find closest frame within 1.5s window
                let bestFrame = null;
                let minDiff = 1.5; 
                
                for (let i = 0; i < videoFrames.length; i++) {{
                    const diff = Math.abs(videoFrames[i].Time - t);
                    if (diff < minDiff) {{
                        minDiff = diff;
                        bestFrame = videoFrames[i];
                    }}
                }}
                
                let videoEmoStr = null;
                if (bestFrame) {{
                    currVideo = bestFrame;
                    videoEmoStr = currVideo.Emotion.toLowerCase();
                    videoBox.innerText = "Video: " + currVideo.Emotion.toUpperCase();
                    videoBox.style.backgroundColor = colors[currVideo.Emotion] || '#333';
                    videoBox.style.color = '#000';
                }} else {{
                    videoBox.innerText = "Video: ...";
                    videoBox.style.backgroundColor = '#222';
                    videoBox.style.color = '#555';
                }}
                
                // 3. Fusion Logic
                if (currAudio && videoEmoStr) {{
                    const audioConf = (activeAudio.Confidence * 100).toFixed(1);
                    const videoConf = (currVideo.Confidence * 100).toFixed(1);
                    if (currAudio === videoEmoStr) {{
                        fusionBox.innerText = `Fused: ${{currAudio.toUpperCase()}} (Audio: ${{audioConf}}%, Video: ${{videoConf}}%)`;
                        fusionBox.style.backgroundColor = '#4CAF50'; // Green match
                        fusionBox.style.color = 'white';
                    }} else {{
                         fusionBox.innerText = `Mixed: Audio ${{currAudio.toUpperCase()}} (${{audioConf}}%) vs Video ${{videoEmoStr.toUpperCase()}} (${{videoConf}}%)`;
                         fusionBox.style.backgroundColor = '#FF9800'; // Orange conflict
                         fusionBox.style.color = 'black';
                    }}
                }} else if (currAudio) {{
                    const audioConf = (activeAudio.Confidence * 100).toFixed(1);
                    fusionBox.innerText = `Audio Only: ${{currAudio.toUpperCase()}} (${{audioConf}}%)`;
                    fusionBox.style.backgroundColor = '#2196F3'; // Blue (Audio only)
                    fusionBox.style.color = 'white';
                }} else if (videoEmoStr) {{
                    const videoConf = (currVideo.Confidence * 100).toFixed(1);
                    fusionBox.innerText = `Video Only: ${{videoEmoStr.toUpperCase()}} (${{videoConf}}%)`;
                    fusionBox.style.backgroundColor = '#9C27B0'; // Purple (Video only)
                    fusionBox.style.color = 'white';
                }} else {{
                    fusionBox.innerText = "Fused: ...";
                    fusionBox.style.backgroundColor = '#333';
                    fusionBox.style.color = '#888';
                }}
            }}

            video.ontimeupdate = updatePlayer;
        </script>
    </div>
    """
    return html

# --- UI ---
st.title("🎥 Interactive Multimodal Emotion Dashboard")
st.markdown("Upload a video to analyze vocal tones (SER) and facial expressions (FER) simultaneously.")

models = load_models()
if not models:
    st.error("No models found. Please check paths.")
    st.stop()

uploaded_file = st.sidebar.file_uploader("1. Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])
uploaded_audio = st.sidebar.file_uploader("2. Upload Audio (Optional, if video has no audio track)", type=['wav', 'mp3'])
num_speakers = st.sidebar.slider("Expected Speakers", 1, 5, 2)
rotation_mode = st.sidebar.radio("Video Rotation (Fix for mobile)", ["None", "90° Clockwise", "180°", "90° Counter-Clockwise"])

if uploaded_file:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) 
    tfile.write(uploaded_file.read())
    tfile.close() # CRITICAL: Close file to ensure it's fully written/flush buffers
    video_path = tfile.name
    
    # Determine Audio Source
    audio_path = video_path
    if uploaded_audio:
        afile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1])
        afile.write(uploaded_audio.read())
        afile.close() # Close audio file too
        audio_path = afile.name
        st.sidebar.info("Using separate audio file for SER analysis.")
    
    # Initialize session state
    if 'df_audio' not in st.session_state:
        st.session_state.df_audio = pd.DataFrame()
    if 'df_video' not in st.session_state:
        st.session_state.df_video = pd.DataFrame()
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    if st.button("Start Analysis"):
        st.session_state.analyzed = False
        with st.status("Processing Multimodal Data...", expanded=True) as status:
            
            # 1. Video Analysis
            st.write("Running Facial Expression Recognition...")
            df_video = pd.DataFrame()
            if 'fer' in models:
                # Pass rotation mode here
                df_video = process_video_fer(video_path, models['fer'], models['face_cascade'], rotation_mode)
                st.session_state.df_video = df_video
                if len(df_video) > 0:
                    st.write(f"✅ Processed {len(df_video)} video frames.")
                else:
                    st.error("❌ Processed 0 frames. Video might be corrupted or format unsupported by OpenCV.")
            
            # 2. Audio Analysis
            st.write("Running Speech Emotion Recognition...")
            df_audio = pd.DataFrame()
            if 'ser' in models:
                # Try to process
                df_audio = process_audio(audio_path, models['ser'], models['ser_classes'], models['ser_type'], num_speakers)
                st.session_state.df_audio = df_audio
                
                if not df_audio.empty:
                    st.write(f"✅ Detected {len(df_audio)} audio segments.")
                else:
                    if not uploaded_audio:
                        st.error("❌ No audio detected! This system lacks FFmpeg to extract audio from video. Please upload the .wav audio file separately in the sidebar.")
                    else:
                        st.warning("Audio processing failed on the uploaded file.")
            
            st.session_state.analyzed = True
            status.update(label="Analysis Complete!", state="complete", expanded=False)

    if st.session_state.analyzed:
        df_audio = st.session_state.df_audio
        df_video = st.session_state.df_video
        
        # --- INTERACTIVE PLAYER (TOP) ---
        st.subheader("Interactive Multimodal Player")
        
        # Check file size before attempting base64 encoding (approx 50MB limit for smooth experience)
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Increased limit to 200MB and rely on browser handling
        if file_size_mb > 200:
            st.warning(f"⚠️ Video is large ({file_size_mb:.1f} MB). Using standard player (No interactive overlay).")
            st.video(video_path)
        else:
            try:
                # Using a taller iframe for mobile-format support
                html_code = get_multimodal_player_html(video_path, df_audio, df_video)
                components.html(html_code, height=750, scrolling=False)
            except Exception as e:
                st.error(f"Could not load interactive player: {e}")
                st.video(video_path)

        # --- VISUALIZATION ---
        st.divider()
        st.subheader("Multimodal Fusion Timeline")
        
        if not df_audio.empty and not df_video.empty:
            # Create a unified timeline
            fig = go.Figure()

            # Add Audio Traces
            fig.add_trace(go.Scatter(
                x=df_audio['Start'], 
                y=[1]*len(df_audio),
                mode='markers',
                marker=dict(size=10, symbol='circle', color=df_audio['Emotion'].map({
                        'neutral': '#D3D3D3', 'calm': '#ADD8E6', 'happy': '#FFD700',
                        'sad': '#4682B4', 'angry': '#FF4500', 'fearful': '#800080',
                        'disgust': '#008000', 'surprised': '#FFA500'
                    })),
                name='Audio Emotion',
                text=df_audio['Emotion'],
                hovertemplate="Audio: %{text}<br>Time: %{x:.2f}s"
            ))

            # Add Video Traces
            fig.add_trace(go.Scatter(
                x=df_video['Time'], 
                y=[2]*len(df_video),
                mode='markers',
                marker=dict(size=8, symbol='square', color=df_video['Emotion'].map({
                        'Neutral': '#D3D3D3', 'Happy': '#FFD700', 'Sad': '#4682B4', 
                        'Angry': '#FF4500', 'Fear': '#800080', 'Disgust': '#008000', 
                        'Surprise': '#FFA500'
                    })),
                name='Video Emotion',
                text=df_video['Emotion'],
                hovertemplate="Video: %{text}<br>Time: %{x:.2f}s"
            ))

            fig.update_layout(
                title="Audio vs Video Emotion Alignment",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2],
                    ticktext=['Audio', 'Video'],
                    range=[0.5, 2.5]
                ),
                xaxis_title="Time (s)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            if not df_audio.empty:
                st.write("#### Audio Detail")
                fig_audio = px.scatter(
                    df_audio, x="Start", y="Emotion", color="Emotion",
                    size="Confidence", hover_data=["Speaker", "Duration"],
                    color_discrete_map={
                        'neutral': '#D3D3D3', 'calm': '#ADD8E6', 'happy': '#FFD700',
                        'sad': '#4682B4', 'angry': '#FF4500', 'fearful': '#800080',
                        'disgust': '#008000', 'surprised': '#FFA500'
                    }
                )
                st.plotly_chart(fig_audio, use_container_width=True)

        with col2:
            if not df_video.empty:
                st.write("#### Video Detail")
                fig_video = px.scatter(
                    df_video, x="Time", y="Emotion", color="Emotion",
                    size="Confidence", 
                    color_discrete_map={
                        'Neutral': '#D3D3D3', 'Happy': '#FFD700', 'Sad': '#4682B4', 
                        'Angry': '#FF4500', 'Fear': '#800080', 'Disgust': '#008000', 
                        'Surprise': '#FFA500'
                    }
                )
                st.plotly_chart(fig_video, use_container_width=True)
