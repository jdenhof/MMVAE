import os
import pandas as pd
import re

class CSVFileLoader:
    def __init__(self, folder_path, filename_pattern, disease, tissue_type):
        self.folder_path = folder_path
        self.filename_pattern = filename_pattern
        self.disease = disease
        self.tissue_type = tissue_type
        self.matching_rows = []
        self.matching_indices = []

    def load_files(self):
        # List all files in the specified folder
        for filename in os.listdir(self.folder_path):
            # Check if the filename matches the pattern
            if re.match(self.filename_pattern, filename):
                file_path = os.path.join(self.folder_path, filename)
                # Load the CSV file
                df = pd.read_csv(file_path)
                # Filter rows matching the specified disease and tissue type
                matches = df[(df['disease'] == self.disease) & (df['tissue'] == self.tissue_type)]
                # Store the matching rows along with the index
                for index, match in matches.iterrows():
                    match_data = match.to_dict()
                    match_data['original_index'] = index  # Store the original index
                    match_data['file_name'] = filename  # Store the filename if needed
                    self.matching_rows.append(match_data)
                    self.matching_indices.append(index)

    def __iter__(self):
        for row in self.matching_rows:
            yield row

def main():
    # Create an instance of the CSVFileLoader
    healthy_loader = CSVFileLoader(folder_path='/active/debruinz_project/human_data/python_data',
                        filename_pattern='chunk10_metadata.csv',  # This will match any .csv file
                        disease='normal',
                        tissue_type='lung')
    
    # Create an instance of the CSVFileLoader
    sick_loader = CSVFileLoader(folder_path='/active/debruinz_project/human_data/python_data',
                        filename_pattern='chunk10_metadata.csv',  # This will match any .csv file
                        disease='cystic fibrosis',
                        tissue_type='lung')

    # Load the CSV files and filter rows
    healthy_loader.load_files()
    sick_loader.load_files()
    
    print(f"Number of healthy samples: {len(healthy_loader.matching_rows)}")
    print(f"Number of sick samples: {len(sick_loader.matching_rows)}")

    # Iterate through the matching rows
    # for row in healthy_loader:
    #     print(row)
        
if __name__ == '__main__':
    main()