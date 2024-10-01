import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import resampy
import soundfile as sf
from moviepy.editor import VideoFileClip
import tensorflow as tf
import tensorflow_hub as hub
import whisper
from transformers import BertTokenizer, BertModel

# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load ResNet model
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # Replace final fully connected layer with identity
resnet.eval()

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_frame_features_from_video(video_path, resnet_model, preprocess_transform):
    """
    Extract features from frames of a video using a pre-trained ResNet model without saving frames.
    
    Parameters:
        video_path (str): Path to the input video file.
        resnet_model (nn.Module): Pre-trained ResNet model.
        preprocess_transform (transforms.Compose): Preprocessing transformations.
        
    Returns:
        features_list (list): List of feature tensors for each frame.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    frame_count = 0
    features_list = []
    
    resnet_model = resnet_model.to(device)  # Move model to device once
    
    with torch.no_grad():  # Disable gradient computation
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Extract features every 50 frames since it's computationally expensive
            if frame_count % 50 == 0:
                # Convert the frame (numpy array) to a PIL image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Preprocess the image
                input_tensor = preprocess_transform(image)
                input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
                
                # Move input batch to the same device as the model
                input_batch = input_batch.to(device)
                
                # Forward pass through the model
                features = resnet_model(input_batch)
                features = features.squeeze().cpu()  # Remove batch dimension and move to CPU
                features_list.append(features)
            
            frame_count += 1
    
    video_capture.release()
    return features_list

def aggregate_features(features_list):
    """
    Aggregate frame features to obtain a video-level feature representation.
    
    Parameters:
        features_list (list): List of feature tensors for each frame.
        
    Returns:
        video_feature (Tensor): Aggregated feature tensor representing the video.
    """
    # Stack features into a tensor
    features_tensor = torch.stack(features_list)  # Shape: [num_frames, feature_dim]
    
    # Compute mean across frames
    video_feature = torch.mean(features_tensor, dim=0)
    
    return video_feature

def extract_audio_from_video(video_path, audio_output_path):
    """
    Extract audio from a video file and save it as a separate audio file.
    
    Parameters:
        video_path (str): Path to the input video file.
        audio_output_path (str): Path to save the extracted audio file (e.g., 'output.wav').
    """
    # Load the video file
    video_clip = VideoFileClip(video_path)
    
    # Get the audio from the video clip
    audio_clip = video_clip.audio
    
    # Write the audio to a file
    audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le')
    
    # Close the clips to release resources
    audio_clip.close()
    video_clip.close()

def extract_audio_features(audio_path):
    """
    Extract audio features from an audio file using the YAMNet model.

    Parameters:
        audio_path (str): Path to the input audio file.

    Returns:
        features (np.ndarray): Audio feature vector from YAMNet model.
    """
    # Read the audio file
    audio_data, sr_orig = sf.read(audio_path)

    # Ensure the audio is mono
    if audio_data.ndim > 1:
        audio_mono = np.mean(audio_data, axis=1)
    else:
        audio_mono = audio_data

    # Resample audio to 16kHz, required for YAMNet
    target_sr = 16000
    audio_resampled = resampy.resample(audio_mono, sr_orig, target_sr)

    # Convert the audio array into the correct shape and type for YAMNet
    audio_waveform = tf.convert_to_tensor(audio_resampled, dtype=tf.float32)

    # Run YAMNet to extract features
    _, embeddings, _ = yamnet_model(audio_waveform)

    # embeddings is of shape (T, 1024), where T is the number of audio frames
    # Average the embeddings over time to get a single feature vector
    features = tf.reduce_mean(embeddings, axis=0).numpy()  # Shape: (1024,)

    return features

def transcribe_audio(audio_path, model_size='base'):
    """
    Transcribe an audio file to text using OpenAI's Whisper model.

    Parameters:
        audio_path (str): Path to the audio file.
        model_size (str): Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        transcript (str): The transcribed text.
    """
    # Load the Whisper model
    model = whisper.load_model(model_size)
    
    # Transcribe the audio file
    result = model.transcribe(audio_path)
    
    # Get the transcribed text
    transcript = result['text']
    
    return transcript

def extract_text_features(text):
    """
    Extract features from text using a pre-trained BERT model.

    Parameters:
        text (str): Input text to process.

    Returns:
        text_feature (Tensor): Feature vector extracted from BERT.
    """
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Get the model outputs
    outputs = bert_model(**inputs)
    
    # Extract the [CLS] token's embedding (batch_size, hidden_size)
    text_feature = outputs.last_hidden_state[:, 0, :]  # [CLS] token is at position 0
    
    return text_feature.squeeze()

class MultimodalTransformer(nn.Module):
    def __init__(self, feature_dim=512, nhead=8, num_layers=2):
        super(MultimodalTransformer, self).__init__()
        # Projection layers to common feature dimension
        self.visual_proj = nn.Linear(2048, feature_dim)
        self.audio_proj = nn.Linear(1024, feature_dim)
        self.text_proj = nn.Linear(768, feature_dim)
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(3, feature_dim)  # 3 modalities
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer (optional, depending on downstream task)
        # self.output_layer = nn.Linear(feature_dim, output_dim)  # Define output_dim appropriately if needed
    
    def forward(self, visual_feature, audio_feature, text_feature):
        # Project features to common dimension
        visual_feature = self.visual_proj(visual_feature.to(device))
        audio_feature = self.audio_proj(audio_feature.to(device))
        text_feature = self.text_proj(text_feature.to(device))
        
        # Stack features to create sequence
        features = torch.stack([visual_feature, audio_feature, text_feature], dim=0)  # Shape: [seq_len, feature_dim]
        
        # Modality embeddings
        modality_ids = torch.tensor([0,1,2], dtype=torch.long).to(device)
        modality_embeds = self.modality_embeddings(modality_ids)
        
        # Add modality embeddings
        features = features + modality_embeds
        
        # Transformer expects input of shape [seq_len, batch_size, feature_dim]
        features = features.unsqueeze(1)  # Add batch dimension
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(features)  # Shape: [seq_len, batch_size, feature_dim]
        
        # Take the mean of the outputs (or use the output corresponding to a [CLS] token if we have one)
        output = transformer_output.mean(dim=0).squeeze(0)  # Shape: [feature_dim]
        
        # Pass through output layer if needed
        # output = self.output_layer(output)
        
        return output


def extract_features(video_path):
    audio_output_path = 'temp_audio.wav'
    
    # Step 1: Extract visual features
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])
    
    # Extract frame features directly from the video
    features_list = extract_frame_features_from_video(video_path, resnet, preprocess)
    
    # Aggregate features
    visual_feature = aggregate_features(features_list)  # Shape: [2048]

    # Extract audio from video
    extract_audio_from_video(video_path, audio_output_path)
    
    # Extract audio features using YAMNet
    audio_feature_np = extract_audio_features(audio_output_path)  # NumPy array of shape (1024,)
    audio_feature = torch.from_numpy(audio_feature_np).float()  # Convert to torch tensor
    
    
    # Transcribe audio to text using Whisper
    transcript = transcribe_audio(audio_output_path, model_size='base')
    
    # Extract text features using BERT
    text_feature = extract_text_features(transcript)  # Shape: [768]
    
    # Remove the temporary audio file
    os.remove(audio_output_path)
    
    # Initialize the Multimodal Transformer
    feature_dim = 512
    multimodal_transformer = MultimodalTransformer(feature_dim=feature_dim, nhead=8, num_layers=2)
    multimodal_transformer.to(device)
    multimodal_transformer.eval()
    
    # Pass the features through the transformer to get the fused feature
    with torch.no_grad():
        fused_feature = multimodal_transformer(visual_feature, audio_feature, text_feature)
    
    # Save the fused feature vector
    torch.save(fused_feature.cpu(), 'fused_feature.pt')

    return fused_feature