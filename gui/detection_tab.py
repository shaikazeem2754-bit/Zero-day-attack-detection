import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import json
import datetime

from blockchain.block_chain import Blockchain

class DetectionTab:
    """Threat detection tab"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main = main_window
        self.frame = ttk.Frame(parent)
        self.blockchain = Blockchain()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the detection tab UI"""
        # Split into left and right panes
        left_frame = ttk.Frame(self.frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(self.frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Left pane - Input features
        input_frame = ttk.LabelFrame(left_frame, text="Packet Features Input", padding=10)
        input_frame.pack(fill='both', expand=True)
        
        # Create scrollable frame for inputs
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add input fields for each feature
        self.input_entries = {}
        columns = self.main.data_processor.columns
        for i, col in enumerate(columns):
            ttk.Label(scrollable_frame, text=f"{col}:").grid(row=i, column=0, padx=5, pady=2, sticky='w')
            entry = ttk.Entry(scrollable_frame, width=15)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.input_entries[col] = entry
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Clear All", command=self.clear_inputs).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Detect Threat", command=self.detect_threat).pack(side='left', padx=5)
        
        # Right pane - Results
        results_frame = ttk.LabelFrame(right_frame, text="Detection Results", padding=10)
        results_frame.pack(fill='both', expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=60, height=35)
        self.results_text.pack(fill='both', expand=True)
    
    def clear_inputs(self):
        """Clear all input fields"""
        for entry in self.input_entries.values():
            entry.delete(0, tk.END)
    
    def detect_threat(self):
        """Perform threat detection with proper error handling"""
        if not self.main.models:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Get input values
            packet_features_dict = {}
            for col in self.main.data_processor.columns:
                value = self.input_entries[col].get()
                if not value:
                    messagebox.showerror("Error", f"Please enter value for {col}")
                    return
                packet_features_dict[col] = value
            
            # Encode the packet using the data processor
            packet_encoded = self.main.data_processor.encode_single_packet(packet_features_dict)
            
            # Scale features
            packet_scaled = self.main.data_processor.scaler.transform(packet_encoded)
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "THREAT DETECTION ANALYSIS\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            
            # Store detection results
            detection_results = {
                'timestamp': datetime.datetime.now().isoformat(),
                'features': packet_features_dict,
                'model_results': {}
            }
            
            overall_threat = "Low"
            
            # Test with each trained model
            for model_name, model in self.main.models.items():
                self.results_text.insert(tk.END, f"📊 {model_name}\n")
                self.results_text.insert(tk.END, "-"*40 + "\n")
                
                try:
                    if model_name == 'Contrastive Autoencoder':
                        # CAE detection
                        result = model.detect_anomalies(packet_scaled)
                        
                        # Handle the result properly - it's a dictionary
                        if isinstance(result, dict):
                            is_anomaly = result['is_anomaly']
                            # Convert to scalar if it's a numpy array
                            if hasattr(is_anomaly, '__len__') and len(is_anomaly) > 0:
                                is_anomaly = is_anomaly[0]
                            error = result['reconstruction_errors']
                            if hasattr(error, '__len__') and len(error) > 0:
                                error = error[0]
                            score = result['anomaly_scores']
                            if hasattr(score, '__len__') and len(score) > 0:
                                score = score[0]
                            threshold = result['threshold']
                        else:
                            # Fallback if result is not a dictionary
                            is_anomaly = False
                            error = 0.0
                            score = 0.0
                            threshold = 0.0
                        
                        if is_anomaly:
                            threat_msg = "⚠️ ANOMALY DETECTED - Potential Zero-Day Attack!"
                            threat_level = "Critical"
                        else:
                            threat_msg = "✓ Normal Pattern Detected"
                            threat_level = "Low"
                        
                        self.results_text.insert(tk.END, f"  Reconstruction Error: {float(error):.4f}\n")
                        self.results_text.insert(tk.END, f"  Anomaly Score: {float(score):.4f}\n")
                        self.results_text.insert(tk.END, f"  Threshold: {float(threshold):.4f}\n")
                        self.results_text.insert(tk.END, f"  Result: {threat_msg}\n")
                        self.results_text.insert(tk.END, f"  Threat Level: {threat_level}\n\n")
                        
                        detection_results['model_results'][model_name] = {
                            'result': threat_msg,
                            'reconstruction_error': float(error),
                            'anomaly_score': float(score),
                            'threshold': float(threshold),
                            'threat_level': threat_level
                        }
                        
                        # Update overall threat
                        if threat_level == "Critical":
                            overall_threat = "Critical"
                        elif threat_level == "High" and overall_threat != "Critical":
                            overall_threat = "High"
                        
                    elif model_name == 'ProtoNet':
                        # ProtoNet detection
                        predictions, distances = model.predict(packet_scaled)
                        
                        # Handle predictions properly
                        if hasattr(predictions, '__len__') and len(predictions) > 0:
                            prediction = predictions[0]
                        else:
                            prediction = predictions
                        
                        # Handle distances properly
                        if isinstance(distances, list) and len(distances) > 0:
                            distance_info = distances[0]
                        else:
                            distance_info = {'min_distance': 0.0, 'closest_label': 0, 'threshold': 0.0}
                        
                        if prediction == -1:
                            threat_msg = "⚠️ ZERO-DAY ATTACK DETECTED - Unknown Pattern!"
                            threat_level = "Critical"
                        else:
                            # Convert prediction to label
                            try:
                                label_name = self.main.data_processor.label_encoder.inverse_transform([prediction])[0]
                                threat_msg = f"✓ Classified as: {label_name}"
                                threat_level = "Medium" if label_name.lower() != 'normal' else "Low"
                            except:
                                threat_msg = f"✓ Classified as: Class {prediction}"
                                threat_level = "Medium" if prediction != 0 else "Low"
                        
                        self.results_text.insert(tk.END, f"  Min Distance: {float(distance_info.get('min_distance', 0)):.4f}\n")
                        self.results_text.insert(tk.END, f"  Closest Class: {distance_info.get('closest_label', 'N/A')}\n")
                        self.results_text.insert(tk.END, f"  Threshold: {float(distance_info.get('threshold', 0)):.4f}\n")
                        self.results_text.insert(tk.END, f"  Result: {threat_msg}\n")
                        self.results_text.insert(tk.END, f"  Threat Level: {threat_level}\n\n")
                        
                        detection_results['model_results'][model_name] = {
                            'result': threat_msg,
                            'min_distance': float(distance_info.get('min_distance', 0)),
                            'closest_class': int(distance_info.get('closest_label', -1)) if distance_info.get('closest_label') is not None else -1,
                            'threat_level': threat_level
                        }
                        
                        # Update overall threat
                        if threat_level == "Critical":
                            overall_threat = "Critical"
                        elif threat_level == "Medium" and overall_threat not in ["Critical", "High"]:
                            overall_threat = "Medium"
                        
                    else:
                        # Traditional ML models
                        prediction = model.predict(packet_scaled)
                        
                        # Handle prediction array
                        if hasattr(prediction, '__len__') and len(prediction) > 0:
                            prediction_value = prediction[0]
                        else:
                            prediction_value = prediction
                        
                        # Get probabilities
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(packet_scaled)
                            if hasattr(proba, '__len__') and len(proba) > 0:
                                confidence = float(max(proba[0]))
                            else:
                                confidence = 0.0
                        else:
                            confidence = 0.0
                        
                        # Convert prediction to label
                        try:
                            result_class = self.main.data_processor.label_encoder.inverse_transform([prediction_value])[0]
                        except:
                            result_class = str(prediction_value)
                        
                        if result_class.lower() == 'normal':
                            threat_msg = "✓ Normal Traffic Detected"
                            threat_level = "Low"
                        else:
                            threat_msg = f"⚠️ ATTACK DETECTED - {result_class}"
                            threat_level = "High"
                        
                        self.results_text.insert(tk.END, f"  Prediction: {result_class}\n")
                        self.results_text.insert(tk.END, f"  Confidence: {confidence:.4f}\n")
                        self.results_text.insert(tk.END, f"  Result: {threat_msg}\n")
                        self.results_text.insert(tk.END, f"  Threat Level: {threat_level}\n\n")
                        
                        detection_results['model_results'][model_name] = {
                            'result': threat_msg,
                            'prediction': result_class,
                            'confidence': confidence,
                            'threat_level': threat_level
                        }
                        
                        # Update overall threat
                        if threat_level == "High" and overall_threat != "Critical":
                            overall_threat = "High"
                        elif threat_level == "Medium" and overall_threat == "Low":
                            overall_threat = "Medium"
                
                except Exception as e:
                    error_msg = f"Error with {model_name}: {str(e)}"
                    self.results_text.insert(tk.END, f"  ❌ {error_msg}\n\n")
                    detection_results['model_results'][model_name] = {
                        'result': f"Error: {str(e)}",
                        'threat_level': "Unknown"
                    }
            
            # Add to blockchain
            block_data = {
                'detection': detection_results,
                'overall_threat_level': overall_threat
            }
            
            block = self.blockchain.add_block(block_data)
            
            # Display blockchain info
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, f"⛓️ BLOCKCHAIN RECORD ADDED\n")
            self.results_text.insert(tk.END, f"  Block Index: {block.index}\n")
            self.results_text.insert(tk.END, f"  Block Hash: {block.hash[:16]}...\n")
            self.results_text.insert(tk.END, f"  Overall Threat Level: {overall_threat}\n")
            self.results_text.insert(tk.END, "="*60 + "\n")
            
            # Save detection to file
            self.save_detection_log(detection_results)
            
            # Show alert for high threats
            if overall_threat in ["High", "Critical"]:
                messagebox.showwarning("⚠️ THREAT ALERT ⚠️", 
                                      f"{overall_threat.upper()} level threat detected!\n\n"
                                      f"Please check the detection results for details.\n"
                                      f"Blockchain record created with hash: {block.hash[:16]}...")
            else:
                messagebox.showinfo("Detection Complete", 
                                   f"Threat detection completed successfully.\n"
                                   f"Overall threat level: {overall_threat}\n"
                                   f"Block stored with hash: {block.hash[:16]}...")
            
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.results_text.insert(tk.END, f"\n❌ ERROR: {error_msg}\n")
            import traceback
            traceback.print_exc()
    
    def save_detection_log(self, detection_results):
        """Save detection results to log file"""
        import os
        log_file = "results/detection_logs.json"
        os.makedirs("results", exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(detection_results)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")