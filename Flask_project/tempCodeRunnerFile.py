

zip_path="/content/archive (2).zip"

extract_path="/content/age_dataset"

with zipfile.ZipFile(zip_path,'r') as zip_ref:
    zip_ref.extractall(extract_path)