#Sign Language Recognition System

This project implements a sign language recognition system that translates hand gestures from videos into text. It utilizes Mediapipe for hand keypoint detection and an Artificial Neural Network (ANN) for classification.

This project requires Python 3.8 or higher.

Ensure the training.json file contains the correct mapping between video IDs and corresponding glosses (words).

This script trains the model using the data in dataset.json and saves the trained model as model2.pth. It also displays training progress, including loss and evaluation metrics, and generates plots for visualization.

Inference: (Currently integrated with training) The main.py script currently performs inference on the training data after each training epoch.

#Model
The model used is an Artificial Neural Network (ANN) implemented with PyTorch.
It consists of linear layers, ReLU activation functions, and Dropout layers for regularization.
The model takes a sequence of 30 frames, each containing 42 hand keypoints (21 per hand), as input.
The output is a probability distribution over the set of possible sign language glosses (words).
Inside the files folder there is a document that explains in detail the model used
# Setup

- `pip -m venv .venv`
- `pip install -r requirements.txt`
- Extract video frames on 25 fps
- Extract the key (30 frames) that provide hand gesture action
- Gather the mediapipe coordinates from each hand in a keypoints array
- Compile all keyspoints from all frames into one array
- Follow `dataset.json` structure for setup

# Running

- Currently infer and training on `python3 main.py`
- Model saved as `model.pth`

# Pending

- 🟨 Try other model architectures CNN etc.
- 🟨 LIVE parsing of frames for live translation
- 🟨 Setup infer pipeline
