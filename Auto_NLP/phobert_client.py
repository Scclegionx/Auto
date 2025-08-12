import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
from datetime import datetime

class PhoBERTSAMClient:
    def __init__(self, root):
        self.root = root
        self.root.title("PhoBERT_SAM - Vietnamese NLP Client")
        self.root.geometry("800x600")
        
        # API Configuration
        self.api_url = "http://localhost:5000"
        
        # Create GUI
        self.create_widgets()
        
        # Check server status
        self.check_server_status()
    
    def create_widgets(self):
        """Tạo các widget cho GUI"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 PhoBERT_SAM - Vietnamese NLP System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Server Status
        status_frame = ttk.LabelFrame(main_frame, text="Server Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Checking server status...")
        self.status_label.pack(side=tk.LEFT)
        
        refresh_btn = ttk.Button(status_frame, text="🔄 Refresh", command=self.check_server_status)
        refresh_btn.pack(side=tk.RIGHT)
        
        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Text", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text input
        self.text_input = scrolledtext.ScrolledText(input_frame, height=4, width=70)
        self.text_input.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack()
        
        predict_btn = ttk.Button(btn_frame, text="🎯 Predict", command=self.predict_text)
        predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        analyze_btn = ttk.Button(btn_frame, text="🔍 Analyze", command=self.analyze_text)
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(btn_frame, text="🗑️ Clear", command=self.clear_input)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        sample_btn = ttk.Button(btn_frame, text="📝 Sample", command=self.load_sample)
        sample_btn.pack(side=tk.LEFT)
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Prediction tab
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="🎯 Prediction")
        
        # Intent result
        intent_frame = ttk.LabelFrame(self.prediction_frame, text="Intent Recognition", padding="5")
        intent_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(intent_frame, text="Detected Intent:").pack(anchor=tk.W)
        self.intent_label = ttk.Label(intent_frame, text="", font=('Arial', 10, 'bold'))
        self.intent_label.pack(anchor=tk.W)
        
        ttk.Label(intent_frame, text="Confidence:").pack(anchor=tk.W)
        self.intent_conf_label = ttk.Label(intent_frame, text="")
        self.intent_conf_label.pack(anchor=tk.W)
        
        # Command result
        command_frame = ttk.LabelFrame(self.prediction_frame, text="Command Processing", padding="5")
        command_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(command_frame, text="Executable Command:").pack(anchor=tk.W)
        self.command_label = ttk.Label(command_frame, text="", font=('Arial', 10, 'bold'))
        self.command_label.pack(anchor=tk.W)
        
        ttk.Label(command_frame, text="Confidence:").pack(anchor=tk.W)
        self.command_conf_label = ttk.Label(command_frame, text="")
        self.command_conf_label.pack(anchor=tk.W)
        
        # Entities result
        entities_frame = ttk.LabelFrame(self.prediction_frame, text="Entity Extraction", padding="5")
        entities_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.entities_text = scrolledtext.ScrolledText(entities_frame, height=8)
        self.entities_text.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="🔍 Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # JSON tab
        self.json_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.json_frame, text="📄 Raw JSON")
        
        self.json_text = scrolledtext.ScrolledText(self.json_frame)
        self.json_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom info
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.mode_label = ttk.Label(bottom_frame, text="Mode: Unknown")
        self.mode_label.pack(side=tk.LEFT)
        
        self.timestamp_label = ttk.Label(bottom_frame, text="")
        self.timestamp_label.pack(side=tk.RIGHT)
    
    def check_server_status(self):
        """Kiểm tra trạng thái server"""
        def check():
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    mode = data.get('mode', 'unknown')
                    self.status_label.config(text=f"✅ Server running - Mode: {mode}")
                    self.mode_label.config(text=f"Mode: {mode}")
                else:
                    self.status_label.config(text="❌ Server error")
            except Exception as e:
                self.status_label.config(text="❌ Server not running")
        
        threading.Thread(target=check, daemon=True).start()
    
    def predict_text(self):
        """Dự đoán text"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text!")
            return
        
        def predict():
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.root.after(0, lambda: self.display_prediction_results(data))
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    
            except Exception as e:
                error_msg = f"Connection error: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=predict, daemon=True).start()
    
    def analyze_text(self):
        """Phân tích text"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text!")
            return
        
        def analyze():
            try:
                response = requests.post(
                    f"{self.api_url}/analyze",
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.root.after(0, lambda: self.display_analysis_results(data))
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    
            except Exception as e:
                error_msg = f"Connection error: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_prediction_results(self, data):
        """Hiển thị kết quả prediction"""
        # Intent
        intent = data.get('intent', 'Unknown')
        intent_conf = data.get('confidence', {}).get('intent', 0.0)
        self.intent_label.config(text=intent)
        self.intent_conf_label.config(text=f"{intent_conf:.3f}")
        
        # Command
        command = data.get('command', 'Unknown')
        command_conf = data.get('confidence', {}).get('command', 0.0)
        self.command_label.config(text=command)
        self.command_conf_label.config(text=f"{command_conf:.3f}")
        
        # Entities
        entities = data.get('entities', [])
        entities_text = ""
        if entities:
            for entity in entities:
                entities_text += f"• {entity.get('text', '')} ({entity.get('label', '')})\n"
        else:
            entities_text = "No entities found"
        
        self.entities_text.delete("1.0", tk.END)
        self.entities_text.insert("1.0", entities_text)
        
        # Update timestamp
        timestamp = data.get('timestamp', '')
        self.timestamp_label.config(text=f"Last update: {timestamp}")
        
        # Update mode
        mode = data.get('mode', 'unknown')
        self.mode_label.config(text=f"Mode: {mode}")
        
        # Raw JSON
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(data, indent=2, ensure_ascii=False))
        
        # Switch to prediction tab
        self.notebook.select(0)
    
    def display_analysis_results(self, data):
        """Hiển thị kết quả analysis"""
        analysis_text = f"""Text Analysis Results:
{'='*50}

Text: {data.get('text', '')}
Length: {data.get('text_length', 0)} characters
Words: {data.get('word_count', 0)} words
Tokens: {data.get('token_count', 0)} tokens

Tokens: {data.get('tokens', [])}

Timestamp: {data.get('timestamp', '')}
"""
        
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert("1.0", analysis_text)
        
        # Raw JSON
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(data, indent=2, ensure_ascii=False))
        
        # Switch to analysis tab
        self.notebook.select(1)
    
    def clear_input(self):
        """Xóa input"""
        self.text_input.delete("1.0", tk.END)
    
    def load_sample(self):
        """Load sample text"""
        samples = [
            "gửi tin nhắn cho mẹ: yêu mẹ nhiều",
            "đặt báo thức lúc 5 giờ chiều",
            "phát nhạc của Sơn Tùng",
            "hôm nay thời tiết Hà Nội thế nào?",
            "tôi bị đau đầu và mệt mỏi"
        ]
        
        # Simple sample selection
        sample = samples[0]  # Use first sample
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", sample)

def main():
    root = tk.Tk()
    app = PhoBERTSAMClient(root)
    root.mainloop()

if __name__ == "__main__":
    main()
