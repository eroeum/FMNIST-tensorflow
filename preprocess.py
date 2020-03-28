import numpy as np

def format_df(df):
    data = df.loc[:, df.columns != 'label'].to_numpy(dtype=np.uint8)
    labels = df['label'].to_numpy()
    splitter = lambda x: np.split(x, 28)
    data = np.array([splitter(r) for r in data], dtype=np.uint8)
    return data, labels

def reshape(images):
    images = images.reshape(images.shape[0], 28, 28, 1)
    return images / 255.0
