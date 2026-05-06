Purpose:
Build a machine learning framework to detect zero-day cyber attacks using deep learning techniques.

Core Idea:
Traditional IDS fail because they rely on known attack signatures, while this project uses representation learning + metric learning to detect unseen attacks.

Workflow:

1. Input network/security data
2. Feature extraction using CAE
3. Classification using ProtoNet
4. Output → Normal / Known attack / Zero-day attack


Tech Stack:

1. Python
2. Deep Learning (PyTorch/TensorFlow)
3. Autoencoders + Meta-learning models


Goal:
Achieve high accuracy with low false positives for detecting unknown cyber threats.
