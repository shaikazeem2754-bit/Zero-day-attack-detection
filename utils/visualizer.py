import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class Visualizer:
    """Visualization utilities for intrusion detection results"""
    
    def __init__(self):
        self.figures = []
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        self.figures.append(fig)
        return fig
    
    def plot_feature_importance(self, feature_names, importances, title="Feature Importance"):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        ax.barh(range(len(sorted_names)), sorted_importances, align='center')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        
        self.figures.append(fig)
        return fig
    
    def plot_reconstruction_errors(self, errors, threshold, title="Reconstruction Errors"):
        """Plot reconstruction errors with threshold"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of errors
        ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Reconstruction Errors')
        ax1.legend()
        
        # Scatter plot
        ax2.scatter(range(len(errors)), errors, c=errors > threshold, 
                   cmap=['green', 'red'], alpha=0.6, s=10)
        ax2.axhline(threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Reconstruction Errors by Sample')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_model_comparison(self, model_names, accuracies, title="Model Performance Comparison"):
        """Plot model performance comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(range(len(model_names)), accuracies, color=colors[:len(model_names)])
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_attack_distribution(self, predictions, title="Attack Distribution"):
        """Plot attack distribution"""
        from collections import Counter
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts = Counter(predictions)
        labels = list(counts.keys())
        values = list(counts.values())
        
        colors = ['#2ecc71' if l == 'normal' else '#e74c3c' for l in labels]
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        ax.set_title(title)
        
        self.figures.append(fig)
        return fig
    
    def plot_prototype_distances(self, distances, labels, title="Prototype Distances"):
        """Plot prototype distances for ProtoNet"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate by class
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            label_indices = labels == label
            ax.scatter(range(len(distances[label_indices])), 
                      distances[label_indices], 
                      label=f'Class {label}', 
                      color=color, alpha=0.6, s=20)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance to Nearest Prototype')
        ax.set_title(title)
        ax.legend()
        
        self.figures.append(fig)
        return fig
    
    def plot_latent_space(self, latent_features, labels, title="Latent Space Visualization"):
        """Plot latent space using t-SNE"""
        from sklearn.manifold import TSNE
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                      label=f'Class {label}', color=color, alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(title)
        ax.legend()
        
        self.figures.append(fig)
        return fig
    
    def plot_training_history(self, history, title="Training History"):
        """Plot training history for neural networks"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        
        # Accuracy (if available)
        if 'accuracy' in history.history:
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.legend()
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def embed_in_tkinter(self, parent, figure):
        """Embed matplotlib figure in tkinter window"""
        canvas = FigureCanvasTkAgg(figure, parent)
        canvas.draw()
        return canvas.get_tk_widget()
    
    def clear_figures(self):
        """Clear all stored figures"""
        for fig in self.figures:
            plt.close(fig)
        self.figures = []