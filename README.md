# Flickr8k-Project1
This project implements a deep learning model that simultaneously embeds images and captions into a shared latent space using a combination of Convolutional Neural Networks (CNNs) for image embedding and Long Short-Term Memory (LSTM) networks for caption embedding.
The model compares the similarity between image and caption embeddings using cosine similarity, and the loss is computed via Mean Squared Error (MSE) between these embeddings.

Project Objectives:
Image Embedding: Utilize CNN to extract feature representations (embeddings) from images.
Caption Embedding: Use LSTM to process and embed captions into the same latent space as the images.
Similarity Calculation: Calculate cosine similarity between image-caption pairs to determine their match.
Loss Function: Use MSE loss to minimize the distance between image and caption embeddings during training.
Accuracy Evaluation: Calculate accuracy based on a predefined threshold for cosine similarity between image-caption pairs.
