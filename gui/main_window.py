import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.data_tab import DataTab
from gui.training_tab import TrainingTab
from gui.detection_tab import DetectionTab
from gui.blockchain_tab import BlockchainTab
from utils.data_processor import DataProcessor

class MainWindow:
    """Main GUI window for Intrusion Detection System"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Intrusion Detection System - AI-Powered Network Security")
        self.root.geometry("1400x800")
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        
        # Setup GUI
        self.setup_menu()
        self.setup_notebook()
        
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        file_menu.add_command(label="Create Sample Dataset", command=self.create_sample_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_notebook(self):
        """Setup notebook tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.data_tab = DataTab(self.notebook, self)
        self.training_tab = TrainingTab(self.notebook, self)
        self.detection_tab = DetectionTab(self.notebook, self)
        self.blockchain_tab = BlockchainTab(self.notebook, self)
        
        # Add tabs to notebook
        self.notebook.add(self.data_tab.frame, text="📊 Data Loading")
        self.notebook.add(self.training_tab.frame, text="🤖 Model Training")
        self.notebook.add(self.detection_tab.frame, text="🔍 Threat Detection")
        self.notebook.add(self.blockchain_tab.frame, text="⛓️ Blockchain Records")
    
    def load_dataset(self):
        """Load dataset from file"""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Load data
                self.data, X, y, y_encoded = self.data_processor.load_data(filepath)
                
                # Preprocess
                X_scaled = self.data_processor.preprocess_data(X, fit_scaler=True)
                
                # Split data
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.data_processor.split_data(X_scaled, y_encoded)
                
                # Update data tab
                self.data_tab.update_data_info(self.data, filepath)
                
                messagebox.showinfo("Success", 
                                   f"Data loaded successfully!\n"
                                   f"Total samples: {len(self.data)}\n"
                                   f"Training samples: {len(self.X_train)}\n"
                                   f"Test samples: {len(self.X_test)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def create_sample_dataset(self):
        """Create sample dataset for testing"""
        try:
            self.data_processor.create_sample_dataset()
            messagebox.showinfo("Success", "Sample dataset created at data/sample_data.csv")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample dataset: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Intrusion Detection System v1.0
        
        An AI-powered network intrusion detection system with:
        • Traditional ML Models (Random Forest, Logistic Regression, KNN)
        • Advanced Deep Learning (Contrastive Autoencoder)
        • Zero-Day Attack Detection (ProtoNet)
        • Blockchain-based Immutable Records
        
        Developed for network security and threat detection.
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()