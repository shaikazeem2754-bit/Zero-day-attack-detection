import tkinter as tk
from tkinter import ttk, scrolledtext

class DataTab:
    """Data loading and preview tab"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main = main_window
        self.frame = ttk.Frame(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the data tab UI"""
        # Title
        title_label = ttk.Label(self.frame, text="Dataset Management", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Data info frame
        info_frame = ttk.LabelFrame(self.frame, text="Dataset Information", padding=10)
        info_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        self.data_info = tk.StringVar(value="No data loaded")
        info_label = ttk.Label(info_frame, textvariable=self.data_info, justify='left')
        info_label.pack()
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(self.frame, text="Data Preview", padding=10)
        preview_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=10, pady=5)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, width=100, height=20)
        self.preview_text.pack(fill='both', expand=True)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.frame, text="Statistics", padding=10)
        stats_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=5, width=80)
        self.stats_text.pack()
        
        # Configure grid weights
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
    
    def update_data_info(self, data, filepath):
        """Update data information display"""
        # Update info
        info_text = f"""
        File: {filepath}
        Total Samples: {len(data)}
        Features: {len(data.columns) - 1 if 'Label' in data.columns else len(data.columns)}
        
        Column Types:
        """
        
        for col in data.columns:
            dtype = data[col].dtype
            info_text += f"  • {col}: {dtype}\n"
        
        self.data_info.set(info_text)
        
        # Update preview with proper formatting
        self.preview_text.delete(1.0, tk.END)
        
        # Display first 20 rows with proper alignment
        preview_df = data.head(20)
        self.preview_text.insert(1.0, preview_df.to_string())
        
        # Update statistics
        self.stats_text.delete(1.0, tk.END)
        
        if 'Label' in data.columns or 'label' in data.columns:
            target_col = 'Label' if 'Label' in data.columns else 'label'
            label_counts = data[target_col].value_counts()
            stats = "Class Distribution:\n"
            for label, count in label_counts.items():
                stats += f"  {label}: {count} ({count/len(data)*100:.1f}%)\n"
            
            stats += f"\nFeature Statistics:\n"
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols[:10]:  # Show first 10 numeric columns
                stats += f"  {col}: mean={data[col].mean():.2f}, std={data[col].std():.2f}\n"
            
            # Show categorical column info
            cat_cols = data.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                stats += f"\nCategorical Columns:\n"
                for col in cat_cols:
                    stats += f"  {col}: {data[col].nunique()} unique values\n"
            
            self.stats_text.insert(1.0, stats)