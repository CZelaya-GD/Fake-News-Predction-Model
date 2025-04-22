import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Loads and preprocesses the data.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """
        Loads the data from the specified path and preprocesses it.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            self.data = self.data.dropna()  # Remove rows with missing values
            return self.data
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} was not found.")
            return None

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        if self.data is None:
            print("Error: Data not loaded. Call load_data() first.")
            return None, None

        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return train_data, test_data
