import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json

from blockchain.block_chain import Blockchain

class BlockchainTab:
    """Blockchain view tab"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main = main_window
        self.frame = ttk.Frame(parent)
        self.blockchain = Blockchain()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the blockchain tab UI"""
        # Top control frame
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Refresh Blockchain", 
                  command=self.refresh_view).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Export Blockchain", 
                  command=self.export_blockchain).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Verify Chain", 
                  command=self.verify_chain).pack(side='left', padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.frame, text="Blockchain Statistics", padding=10)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=80)
        self.stats_text.pack()
        
        # Blockchain display
        display_frame = ttk.LabelFrame(self.frame, text="Blockchain Records", padding=10)
        display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.display_text = scrolledtext.ScrolledText(display_frame, width=120, height=30)
        self.display_text.pack(fill='both', expand=True)
        
        # Initial refresh
        self.refresh_view()
    
    def refresh_view(self):
        """Refresh blockchain display"""
        self.display_text.delete(1.0, tk.END)
        
        # Get statistics
        stats = self.blockchain.get_statistics()
        
        # Update statistics display
        self.stats_text.delete(1.0, tk.END)
        stats_display = f"""
        Total Blocks: {stats['total_blocks']}
        Total Detections: {stats['total_detections']}
        Chain Valid: {'✓ YES' if stats['is_valid'] else '✗ NO'}
        """
        
        if stats['attack_counts']:
            stats_display += "\nThreat Distribution:\n"
            for threat, count in stats['attack_counts'].items():
                stats_display += f"  {threat}: {count}\n"
        
        self.stats_text.insert(1.0, stats_display)
        
        # Display blocks
        for block in self.blockchain.chain:
            self.display_text.insert(tk.END, "="*80 + "\n")
            self.display_text.insert(tk.END, f"Block #{block.index}\n")
            self.display_text.insert(tk.END, f"Timestamp: {block.timestamp}\n")
            self.display_text.insert(tk.END, f"Hash: {block.hash}\n")
            self.display_text.insert(tk.END, f"Previous Hash: {block.previous_hash}\n")
            
            if block.data != "Genesis Block - Intrusion Detection System":
                if 'detection' in block.data:
                    detection = block.data['detection']
                    self.display_text.insert(tk.END, f"\nDetection Results:\n")
                    self.display_text.insert(tk.END, f"  Time: {detection['timestamp']}\n")
                    self.display_text.insert(tk.END, f"  Overall Threat: {block.data.get('overall_threat_level', 'N/A')}\n")
                    
                    for model_name, result in detection['model_results'].items():
                        self.display_text.insert(tk.END, f"\n  Model: {model_name}\n")
                        self.display_text.insert(tk.END, f"    Result: {result['result']}\n")
                        self.display_text.insert(tk.END, f"    Threat Level: {result['threat_level']}\n")
                        
                        if 'confidence' in result:
                            self.display_text.insert(tk.END, f"    Confidence: {result['confidence']:.4f}\n")
                        if 'reconstruction_error' in result:
                            self.display_text.insert(tk.END, f"    Reconstruction Error: {result['reconstruction_error']:.4f}\n")
            
            self.display_text.insert(tk.END, "\n")
        
        self.display_text.see(tk.END)
    
    def verify_chain(self):
        """Verify blockchain integrity"""
        is_valid = self.blockchain.is_chain_valid()
        if is_valid:
            messagebox.showinfo("Blockchain Verification", 
                               "Blockchain is valid and has not been tampered with!")
        else:
            messagebox.showerror("Blockchain Verification", 
                                "WARNING: Blockchain has been tampered with!")
    
    def export_blockchain(self):
        """Export blockchain to file"""
        from tkinter import filedialog
        import json
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                chain_data = []
                for block in self.blockchain.chain:
                    chain_data.append({
                        'index': block.index,
                        'timestamp': block.timestamp,
                        'data': block.data,
                        'previous_hash': block.previous_hash,
                        'hash': block.hash,
                        'nonce': block.nonce
                    })
                
                with open(filename, 'w') as f:
                    json.dump(chain_data, f, indent=2)
                
                messagebox.showinfo("Export Successful", 
                                   f"Blockchain exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))