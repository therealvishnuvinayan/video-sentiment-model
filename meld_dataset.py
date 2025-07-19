from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


if __name__ == '__main__':
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete' )
