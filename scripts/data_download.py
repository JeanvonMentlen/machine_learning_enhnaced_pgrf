import os
import requests
from tqdm import tqdm
import zipfile
import sys


def download_file(url, filename):
    """
    Download a file with a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f'Downloading {filename}'
    )

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def create_data_directory():
    """
    Create data directory structure if it doesn't exist
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # Create main data directory and subdirectories
    for subdir in ['train', 'validate', 'test']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    return base_dir


def main():
    # Will be updated after Zenodo upload
    ZENODO_URL = "YOUR_ZENODO_URL_HERE"

    try:
        # Create data directory structure
        data_dir = create_data_directory()

        # Download zip file
        zip_path = os.path.join(data_dir, 'pgrf_data.zip')
        print("Downloading PGRF data...")
        download_file(ZENODO_URL, zip_path)

        # Extract files
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up
        os.remove(zip_path)
        print("Download and extraction complete!")

        # Verify directory structure
        for subdir in ['train', 'validate', 'test']:
            if not os.path.exists(os.path.join(data_dir, subdir)):
                print(f"Warning: {subdir} directory was not created properly")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()