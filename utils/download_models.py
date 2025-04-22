import gdown
import zipfile
import os

def download_and_extract_models():
    zip_file = "model_pkl.zip"
    extracted_folder = os.path.join(os.getcwd(), "model_pkl")
    file_id = "1b5T7Ocs50pn4_75Io0Dk-uaUDLEI1HOn"
    url = f"https://drive.google.com/uc?id={file_id}"

    if os.path.exists(extracted_folder) and os.path.isdir(extracted_folder):
        print("Model folder already exists, skipping download.")
        return

    try:
        if not os.path.exists(zip_file):
            print("📥 Downloading model_pkl.zip...")
            gdown.download(url, zip_file, quiet=False, fuzzy=True)

        print("Extracting model files...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

        print("Extraction completed.")

    except zipfile.BadZipFile:
        print("Corrupted ZIP file.")
    except Exception as e:
        print(f"Unexpected error during extraction: {e}")

    if os.path.exists(zip_file):
        os.remove(zip_file)
        print("Zip file removed after extraction.")
