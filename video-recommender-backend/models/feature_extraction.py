import numpy as np
import cv2
import librosa
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import speech_recognition as sr
import subprocess
import tempfile
import os

def extract_features(video_path):
    # Extract visual features
    visual_features = extract_visual_features(video_path)

    # Extract audio features
    audio_features = extract_audio_features(video_path)

    # Extract text features
    text_features = extract_text_features(video_path)

    # Combine features
    features = {
        'visual': visual_features,
        'audio': audio_features,
        'text': text_features
    }

    return features

def extract_visual_features(video_path):
    # Load pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)
    resnet.eval()

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    # Process frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    features = []
    for frame in frames[::30]:  # Sample every 30 frames
        input_tensor = transform(frame)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = resnet(input_batch)
        features.append(output.numpy())

    visual_features = np.mean(features, axis=0)
    return visual_features.tolist()

def extract_audio_features(video_path):
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name

    # Extract audio from video using subprocess
    command = [
        'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
        audio_path, '-y'
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load pre-trained VGGish model and extract features
    vggish_features = extract_vggish_features(audio_path)

    # Remove the temporary audio file
    os.remove(audio_path)
    return vggish_features

def extract_text_features(video_path):
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name

    # Extract audio from video using subprocess
    command = [
        'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
        audio_path, '-y'
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    os.remove(audio_path)

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()

    # Tokenize and encode text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    text_features = outputs.last_hidden_state.mean(dim=1).numpy()
    return text_features.tolist()

def extract_vggish_features(audio_path):
    # Implement or import your VGGish feature extraction here
    # This is a placeholder function
    features = np.random.rand(128)  # Example feature vector
    return features.tolist()
