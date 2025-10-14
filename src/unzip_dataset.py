import os, zipfile
from src.config import ZIP_PATH, EXTRACT_DIR

def unzip_dataset():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    # quick guard: if already extracted, re-use
    if os.path.exists(EXTRACT_DIR) and any([True for _ in os.scandir(EXTRACT_DIR)]):
        print(f'Using existing extracted folder: {EXTRACT_DIR}')
        return EXTRACT_DIR
    print('Unzipping', ZIP_PATH, '->', EXTRACT_DIR)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print('Unzip complete.')
    return EXTRACT_DIR

if __name__ == '__main__':
    unzip_dataset()
