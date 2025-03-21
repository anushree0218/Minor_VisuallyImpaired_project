import gdown
import os
import zipfile

def download_and_extract(url: str, output_dir: str = "data/input_videos") -> str:
    """Download video from Google Drive link and extract if zipped"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract file ID from URL
    file_id = url.split('/d/')[1].split('/')[0] if 'drive.google.com' in url else url
    
    try:
        # Download the file
        output_path = os.path.join(output_dir, 'downloaded_content')
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        
        # Check if file is zip and extract
        if zipfile.is_zipfile(output_path):
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(output_path)
            extracted_files = [f for f in os.listdir(output_dir) if not f.startswith('.')]
            return os.path.join(output_dir, extracted_files[0])
            
        return output_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")