#!/usr/bin/env python3
"""
Intrusion Detection System - Main Entry Point
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow

def main():
    """Main entry point for the application"""
    print("="*60)
    print("Intrusion Detection System v1.0")
    print("Starting application...")
    print("="*60)
    
    # Create and run application
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()