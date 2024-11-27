import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
import glob
import cv2
import zipfile
from tqdm import tqdm
import re
from collections import Counter

# creating directory to save Kaggle Json file and giving permissions for the file
import os
os.makedirs('/root/.kaggle',exist_ok= True)

# Move the kaggle.json file to the correct location
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')

# Set permissions for the kaggle.json file
os.chmod('/root/.kaggle/kaggle.json', 0o600)

!pip install kaggle

#Download ZipFile form Kaggle Datasets
!kaggle datasets download -d adityajn105/flickr8k

# Specify the path to the zip file
zip_file_path = '/content/flickr8k.zip'

unzip_file_path = '/content/flickr8k'

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_file_path)

print(f"Files extracted to: {unzip_file_path}")
os.listdir('/content/flickr8k')

images = glob.glob('/content/flickr8k/Images/*.jpg')
image_filenames = [os.path.basename(path) for path in images]
print(len(image_filenames))

# Mean And STD calculation of given dataset
def calculate_mean_std(image_paths):
    # Initialize variables to calculate mean and std
    total_pixels = 0
    channel_sum = np.zeros(3)  # For R, G, B
    channel_sum_squared = np.zeros(3)  # For R^2, G^2, B^2

    for image_path in tqdm(image_paths, desc="Processing images"):
        # Load the image in RGB format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Resize to a common size

        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Update sums for mean and variance calculation
        total_pixels += image.shape[0] * image.shape[1]
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(image ** 2, axis=(0, 1))

    # Calculate mean and std
    mean = channel_sum / total_pixels
    variance = (channel_sum_squared / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    return mean, std
M,S = calculate_mean_std(images)

print(M,S)

M = [0.45754746, 0.44561315, 0.40343669]
S = [0.27459249, 0.26735015, 0.2818728 ]

#Defining a to display images
def display_images(pictures,preprocess_imgs):
  plt.figure(figsize=(10,5))
  if pictures :
    for i,image_path in enumerate(pictures[:10]):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.subplot(2,5,i+1)
        plt.imshow(image)
        plt.axis('off')  # Hide axis
  elif preprocess_imgs is not None:
    for i, image_data in enumerate(preprocess_imgs[:10]):
            image_data = np.transpose(image_data, (1, 2, 0))  # Transpose preprocessed image data
            image = (image_data * 255).astype(np.uint8)  # Assuming image_data is normalized
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)
            plt.axis('off')  # Hide axis

  plt.show()
  plt.tight_layout()

display_images(images,None)

#image preprocessing
def preprocess_image_opencv(image_path,mean,std):

    # Step 1: Load the image in RGB format
    image = cv2.imread(image_path)  # Load the image (BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Step 2: Resize the image to (224, 224)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Step 3: Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Step 4: Normalize pixel values using mean and std
    image = (image - mean) / std

    # Step 5: Transpose to (3, 224, 224) format (channels first)
    image = np.transpose(image, (2, 0, 1))

    return image

preprocessed_images = []  # List to store preprocessed images

for image_path in images:
    preprocessed_image = preprocess_image_opencv(image_path,M,S)
    preprocessed_images.append(preprocessed_image)


print(f"Number of preprocessed images: {len(preprocessed_images)}")
print(f"Shape of first preprocessed image: {preprocessed_images[0].shape}")

def preprocess_captions(caps):
    print(len(caps))

    # Step 1: Tokenize and clean captions
    def clean_caption(caption):
      caption = caption.lower()  # Convert to lowercase
      caption = re.sub(r"[^a-z0-9\s]", "", caption)  # Remove punctuation
      caption = " ".join(caption.split())  # Remove extra spaces
      words = caption.split()  # Tokenize caption into words
      unique_words = sorted(set(words))  # Collect unique words and sort alphabetically
      return unique_words

    tokenized_captions = [clean_caption(caption) for caption in captions]


    # Step 2: Build vocabulary
    word_counts = Counter(word for caption in tokenized_captions for word in caption)
    vocab = [word for word, count in word_counts.items()]


    # Step 3: Add special tokens
    special_tokens = ["<bos>", "<eos>", "<unk>","<pad>",]
    vocab = special_tokens + vocab
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Step 4: Encode captions with special tokens
    def encode_caption(caption):
        encoded = [word_to_index.get(word, word_to_index["<unk>"]) for word in caption]
        encoded = [word_to_index["<bos>"]] + encoded + [word_to_index["<eos>"]]
        return encoded

    encoded_captions = [encode_caption(caption) for caption in tokenized_captions]

    def pad_captions(captions, max_length):
      encoded_captions = []
      for caption in captions:
          if len(caption) < max_length:
              padded_cap = caption + [word_to_index.get('<pad>')] * (max_length - len(caption))
              encoded_captions.append(padded_cap)
          else:
              encoded_captions.append(caption[:max_length])
      return encoded_captions

    equal_encoded_captions = pad_captions(encoded_captions, 16)

    return word_to_index, equal_encoded_captions


captions_df = pd.read_csv('/content/flickr8k/captions.txt')
captions = captions_df['caption'].tolist()

word_index, equal_encoded_captions = preprocess_captions(captions)

encoded_16 = 0
others = 0
for cap in equal_encoded_captions:
  if len(cap) == 16:
    encoded_16+=1
  else:
    others+=1
print("encoded to Sizeof 16:",encoded_16,"\t others:",others)

len(word_index)

captions_df['captions']=equal_encoded_captions

images_df = pd.DataFrame({'image': image_filenames, 'Preprocessed_images': preprocessed_images})

final_df = pd.merge(images_df, captions_df, on='image', how='inner')

final_df.drop("image",axis=1,inplace=True)
final_df.drop("caption",axis=1,inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the DataFrame into train and validation sets
train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)


import torch
from torch.utils.data import Dataset, DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe):

        self.dataframe = dataframe

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):

        # Get the row corresponding to the index
        row = self.dataframe.iloc[idx]

        # Extract preprocessed image and caption
        image = row["Preprocessed_images"]  # List of arrays
        caption = row["captions"]  # List of integers (encoded captions)

        # Convert them to torch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)  # Ensure it's float32 for model input
        caption_tensor = torch.tensor(caption, dtype=torch.long)  # Ensure it's long for tokenized captions

        return image_tensor, caption_tensor


# Create dataset objects
train_dataset = ImageCaptionDataset(train_df)
val_dataset = ImageCaptionDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for batch_idx, (images, captions) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Image Tensor Shape: {images.shape}")
    print(f"Caption Tensor Shape: {captions.shape}")
    print(f"No.of batches in train_df:{len(train_loader)}\n")

    print("Shape of single image and caption pair:\n")
    print(f"Single Image Shape: {images[0].shape}")
    print(f"Single Caption Shape: {captions[0].shape}")
    print(images[0])
    print(captions[0])

    # Break after inspecting the first batch (optional)
    break

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNImageEmbedding(nn.Module):
    def __init__(self, output_dim):
        super(CNNImageEmbedding, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 56 * 56, 512)

        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Apply conv1, ReLU activation, and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.3, training=self.training)


        x = x.view(x.size(0), -1)  # Flatten to Batch size x (128 * 28 * 28)
        x = F.relu(self.fc1(x))
        # to get the final embedding
        x = self.fc2(x)

        return x


class LSTMCaptionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMCaptionEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        x = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(x)
        caption_embedding = hn[-1]  # Use the hidden state from the last time step
        caption_embedding = self.fc(caption_embedding)
        return caption_embedding


class ImageCaptionModel(nn.Module):
    def __init__(self, cnn, lstm, output_dim):
        super(ImageCaptionModel, self).__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.output_dim = output_dim

    def forward(self, images, captions):
        # Get image embeddings from the CNN
        image_embeddings = self.cnn(images)

        # Get caption embeddings from the LSTM
        caption_embeddings = self.lstm(captions)

        # Calculate the MSE between the image and caption embeddings
        loss = nn.MSELoss()(image_embeddings, caption_embeddings)
        return loss, image_embeddings, caption_embeddings


import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity
def calculate_cosine_similarity(image_embeddings, caption_embeddings):
    # Cosine similarity requires reshaping the embeddings to (batch_size, -1)
    # If your embeddings are 2D, flatten them to 1D
    image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
    caption_embeddings = caption_embeddings.view(caption_embeddings.size(0), -1)

    # Compute cosine similarity
    similarity = F.cosine_similarity(image_embeddings, caption_embeddings)
    return similarity

# Function to calculate accuracy
def calculate_accuracy(image_embeddings, caption_embeddings, threshold=0.7):
    # Calculate cosine similarity between image and caption embeddings
    similarity = calculate_cosine_similarity(image_embeddings, caption_embeddings)

    # Determine if the similarity is above the threshold
    correct = (similarity > threshold).float()

    # Accuracy is the fraction of correct predictions
    accuracy = correct.sum() / correct.size(0)
    return accuracy

# Training loop with accuracy calculation
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Hyperparameters
output_dim = 512  # Desired embedding size for both image and caption
vocab_size = len(word_index)  # Size of the vocabulary
embedding_dim = 256  # Embedding size for the words
hidden_dim = 512  # LSTM hidden size
learning_rate = 0.001

# Check if GPU is available, otherwise default to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the CNN, LSTM, and combined model
cnn_model = CNNImageEmbedding(output_dim).to(device)
lstm_model = LSTMCaptionEmbedding(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
model = ImageCaptionModel(cnn_model, lstm_model, output_dim).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# GradScaler for mixed precision training
scaler = GradScaler()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for images, captions in train_loader:
        # Move the batch to the GPU
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Mixed precision training with autocast
        with autocast():
            # Forward pass: get image and caption embeddings
            loss, image_embeddings, caption_embeddings = model(images, captions)

            # Calculate the MSE loss between the image and caption embeddings
            loss = F.mse_loss(image_embeddings, caption_embeddings)

        # Backward pass: compute gradients and update weights using scaler
        scaler.scale(loss).backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step the optimizer (scaled gradients)
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        accuracy = calculate_accuracy(image_embeddings, caption_embeddings)

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {avg_accuracy * 100}%")


import torch
import torch.nn.functional as F

# Function to calculate cosine similarity
def calculate_cosine_similarity(image_embeddings, caption_embeddings):
    image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
    caption_embeddings = caption_embeddings.view(caption_embeddings.size(0), -1)
    similarity = F.cosine_similarity(image_embeddings, caption_embeddings)
    return similarity

# Function to calculate accuracy
def calculate_accuracy(image_embeddings, caption_embeddings, threshold=0.7):
    similarity = calculate_cosine_similarity(image_embeddings, caption_embeddings)
    correct = (similarity > threshold).float()  # 1 if similarity > threshold, else 0
    accuracy = correct.sum() / correct.size(0)  # Mean accuracy for the batch
    return accuracy

# Evaluate on the validation set
def evaluate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # Disable gradient calculation for inference
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass: get image and caption embeddings
            loss, image_embeddings, caption_embeddings = model(images, captions)

            # Calculate the MSE loss between the image and caption embeddings
            loss = F.mse_loss(image_embeddings, caption_embeddings)

            # Calculate accuracy
            accuracy = calculate_accuracy(image_embeddings, caption_embeddings)

            total_loss += loss.item()
            total_accuracy += accuracy.item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    return avg_loss, avg_accuracy

# Validation phase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_loss, val_accuracy = evaluate(model, val_loader, device)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy * 100}%")
