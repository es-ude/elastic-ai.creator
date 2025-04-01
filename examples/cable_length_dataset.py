import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Define a custom PyTorch Dataset
class CableLengthDataset(Dataset):
    def __init__(self, csv_path: str):
        # Load the CSV file
        file_path = csv_path  # Update with your file path
        data = pd.read_csv(file_path)

        # Preprocess the data: encode the target column and extract features/targets
        label_encoder = LabelEncoder()
        data['Spannung'] = label_encoder.fit_transform(data['Spannung'])

        # Separate features and targets
        features = data.iloc[:, 1:].values
        targets = data['Spannung'].values

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)  # For classification tasks
        #print(torch.unique(self.targets))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


if __name__ == '__main__':
    dataset = CableLengthDataset('fixed_current_training_data.csv')
