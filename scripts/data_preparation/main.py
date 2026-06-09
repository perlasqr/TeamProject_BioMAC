from data_processor import DataProcessor

# 1. Define your root directory
root = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject" 

# 2. Instantiate for one of your databases
db_processor = DataProcessor(root, "DB5")

# 3. Trigger the processing loop
# This will trigger the print statements to confirm it's finding participants and files
db_processor.process_database()