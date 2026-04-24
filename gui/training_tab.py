import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import numpy as np

from models.traditional_models import TraditionalModels
from models.contrastive_autoencoder import ContrastiveAutoencoder
from models.protonet import ProtoNet

class TrainingTab:
    """Model training tab"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main = main_window
        self.frame = ttk.Frame(parent)
        self.traditional_models = TraditionalModels()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the training tab UI"""
        # Title
        title_label = ttk.Label(self.frame, text="Model Training", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Model selection frame
        selection_frame = ttk.LabelFrame(self.frame, text="Select Models to Train", padding=10)
        selection_frame.grid(row=1, column=0, sticky='nw', padx=10, pady=5)
        
        # Traditional models
        ttk.Label(selection_frame, text="Traditional Models:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        
        self.train_rf = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="Random Forest", variable=self.train_rf).grid(row=1, column=0, sticky='w', padx=20)
        
        self.train_lr = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="Logistic Regression", variable=self.train_lr).grid(row=2, column=0, sticky='w', padx=20)
        
        self.train_knn = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="K-Nearest Neighbors", variable=self.train_knn).grid(row=3, column=0, sticky='w', padx=20)
        
        # Advanced models
        ttk.Label(selection_frame, text="Advanced Models:", font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky='w', pady=5)
        
        self.train_cae = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="Contrastive Autoencoder", variable=self.train_cae).grid(row=1, column=1, sticky='w', padx=20)
        
        self.train_protonet = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="ProtoNet (Zero-Day Detection)", variable=self.train_protonet).grid(row=2, column=1, sticky='w', padx=20)
        
        # Training button
        self.train_button = ttk.Button(selection_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.progress = ttk.Progressbar(selection_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.frame, text="Training Results", padding=10)
        results_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=30)
        self.results_text.pack(fill='both', expand=True)
        
        # Configure grid weights
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
    
    def start_training(self):
        """Start training selected models"""
        if self.main.X_train is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        # Disable button and start progress
        self.train_button.config(state='disabled')
        self.progress.start()
        self.results_text.delete(1.0, tk.END)
        
        # Run training in separate thread
        thread = threading.Thread(target=self.train_models)
        thread.start()
    
    def train_models(self):
        """Train selected models"""
        try:
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "TRAINING STARTED\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            self.results_text.see(tk.END)
            
            # Train traditional models
            if self.train_rf.get():
                self.results_text.insert(tk.END, "Training Random Forest...\n")
                self.results_text.see(tk.END)
                rf, acc, report = self.traditional_models.train_random_forest(
                    self.main.X_train, self.main.y_train,
                    self.main.X_test, self.main.y_test
                )
                self.main.models['Random Forest'] = rf
                self.results_text.insert(tk.END, f"✓ Random Forest Accuracy: {acc:.4f}\n\n")
                self.results_text.see(tk.END)
            
            if self.train_lr.get():
                self.results_text.insert(tk.END, "Training Logistic Regression...\n")
                self.results_text.see(tk.END)
                lr, acc, report = self.traditional_models.train_logistic_regression(
                    self.main.X_train, self.main.y_train,
                    self.main.X_test, self.main.y_test
                )
                self.main.models['Logistic Regression'] = lr
                self.results_text.insert(tk.END, f"✓ Logistic Regression Accuracy: {acc:.4f}\n\n")
                self.results_text.see(tk.END)
            
            if self.train_knn.get():
                self.results_text.insert(tk.END, "Training KNN...\n")
                self.results_text.see(tk.END)
                knn, acc, report = self.traditional_models.train_knn(
                    self.main.X_train, self.main.y_train,
                    self.main.X_test, self.main.y_test
                )
                self.main.models['KNN'] = knn
                self.results_text.insert(tk.END, f"✓ KNN Accuracy: {acc:.4f}\n\n")
                self.results_text.see(tk.END)
            
            # Train advanced models
            if self.train_cae.get():
                self.results_text.insert(tk.END, "Training Contrastive Autoencoder...\n")
                self.results_text.see(tk.END)
                input_dim = self.main.X_train.shape[1]
                cae = ContrastiveAutoencoder(input_dim, latent_dim=32)
                history = cae.fit_with_threshold(self.main.X_train, epochs=50)
                self.main.models['Contrastive Autoencoder'] = cae
                self.results_text.insert(tk.END, f"✓ Contrastive Autoencoder trained\n")
                self.results_text.insert(tk.END, f"  Reconstruction threshold: {cae.reconstruction_threshold:.4f}\n\n")
                self.results_text.see(tk.END)
            
            if self.train_protonet.get():
                self.results_text.insert(tk.END, "Training ProtoNet for Zero-Day Detection...\n")
                self.results_text.see(tk.END)
                protonet = ProtoNet(n_prototypes_per_class=10)
                protonet.fit(self.main.X_train, self.main.y_train)
                self.main.models['ProtoNet'] = protonet
                self.results_text.insert(tk.END, f"✓ ProtoNet trained with {len(protonet.prototypes)} prototypes\n\n")
                self.results_text.see(tk.END)
            
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "TRAINING COMPLETED SUCCESSFULLY!\n")
            self.results_text.insert(tk.END, "="*60 + "\n")
            
            messagebox.showinfo("Success", "Model training completed!")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"\nERROR: {str(e)}\n")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        
        finally:
            # Re-enable button and stop progress
            self.train_button.config(state='normal')
            self.progress.stop()