"""
Main entry point for processing a selected database.

Usage:
    1. Update `root` to the local project repository path.
    2. Set `db_name` to the database that should be processed (e.g. "DB3").
    3. Run:
           python main.py

The script runs the DataProcessor pipeline and exports the processing
report and four ML feature files to <root>/DATA/.
"""

from data_processor import DataProcessor
import matplotlib.pyplot as plt

# Path to the root directory of the project.
# Update this path when running the code on another computer.
root = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject\GitLab" 

# Database to process.
db_name = "DB2"

processor = DataProcessor(root, db_name)

# Trigger the database processing and export.
if processor.participants:
    processor.export_database_results()
else:
    print("No participants to process.")