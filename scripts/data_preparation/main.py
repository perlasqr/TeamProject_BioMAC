from data_processor import DataProcessor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

## ------- MAIN ------- ##
root = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject" 
db_name = "DB5"

processor = DataProcessor(root, db_name)

# Trigger the entire database pipeline and export
if processor.participants:
    processor.export_database_results_v2()
else:
    print("No participants to process.")