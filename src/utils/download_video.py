import gdown
import os
import zipfile
import filetype
from moviepy import VideoFileClip


def convert_to_mp4(input_path: str, output_path: str) -> None:
    """Convert a video file to MP4 format using moviepy."""
    clip = None
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264', logger=None)
    finally:
        if clip:
            clip.close()

def download_and_convert(url: str, output_dir: str = "data/input_videos") -> str:
    """Download a file, convert to MP4 if needed, and return the MP4 path."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract Google Drive file ID
    file_id = url.split('/d/')[1].split('/')[0] if 'drive.google.com' in url else url
    
    try:
        # Download with original filename
        downloaded_path = gdown.download(
            f"https://drive.google.com/uc?id={file_id}", 
            output=os.path.join(output_dir, "temp_file"),  # Temporary name
            quiet=False
        )
        print(f"Downloaded to: {downloaded_path}")

        # Check if ZIP and extract
        if zipfile.is_zipfile(downloaded_path):
            with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(downloaded_path)
            print("ZIP extracted. Checking for video files...")
            
            # Process extracted files
            for f in os.listdir(output_dir):
                file_path = os.path.join(output_dir, f)
                if os.path.isfile(file_path):
                    final_path = process_file(file_path, output_dir)
                    if final_path:
                        return final_path
            raise ValueError("No convertible video found in ZIP")

        # Process directly downloaded file
        return process_file(downloaded_path, output_dir)

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")

def process_file(file_path: str, output_dir: str) -> str:
    """Process an individual file: detect type, convert to MP4 if needed."""
    # Detect MIME type
    kind = filetype.guess(file_path)
    if not kind:
        raise ValueError("Unrecognized file type")

    # Handle video files
    if kind.mime.startswith('video/'):
        output_path = os.path.join(output_dir, f"output_{os.path.basename(file_path)}.mp4")
        
        if kind.mime == 'video/mp4':
            # Just rename if extension is missing
            if not file_path.lower().endswith('.mp4'):
                os.rename(file_path, output_path)
                print(f"Renamed to MP4: {output_path}")
            else:
                output_path = file_path
        else:
            # Convert to MP4
            print(f"Converting {kind.extension.upper()} to MP4...")
            convert_to_mp4(file_path, output_path)
            os.remove(file_path)
        
        return output_path

    raise ValueError(f"Not a video file: {kind.mime}")

if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1PVLJsrwGUtgnH0iIIfQGnxiYks8hNRVe/view?usp=drivesdk"
    output_dir = "data/input_videos"
    
    try:
        mp4_path = download_and_convert(url, output_dir)
        print(f"Success! MP4 file at: {mp4_path}")
    except Exception as e:
        print(f"Error: {e}")