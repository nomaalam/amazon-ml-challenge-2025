# src/utils.py
import os
import requests
from tqdm import tqdm
import numpy as np

def download_images(df, folder_name):
    """
    Downloads images from URLs in a DataFrame and saves them to a specified folder.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'sample_id' and 'image_link' columns.
        folder_name (str): The name of the folder to save images in (e.g., 'train' or 'test').
    """
    # Create the main images directory and the specific sub-folder if they don't exist
    base_image_dir = 'images'
    specific_image_dir = os.path.join(base_image_dir, folder_name)
    os.makedirs(specific_image_dir, exist_ok=True)

    print(f"Starting image download for '{folder_name}'...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Downloading {folder_name} images"):
        image_url = row['image_link']
        sample_id = row['sample_id']
        image_extension = os.path.splitext(image_url)[1].split('?')[0]
        if not image_extension:
            image_extension = '.jpg' # Default extension
            
        image_filename = f"{sample_id}{image_extension}"
        image_path = os.path.join(specific_image_dir, image_filename)

        # Download the image only if it doesn't already exist
        if not os.path.exists(image_path):
            try:
                response = requests.get(image_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                # else:
                #     print(f"Warning: Failed to download image for {sample_id}. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                # print(f"Warning: Error downloading image for {sample_id}. Error: {e}")
                pass # Suppress warnings for cleaner output
    print("Image download complete.")


def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true (np.array): Array of true values.
        y_pred (np.array): Array of predicted values.
        
    Returns:
        float: The SMAPE score.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-8
    ratio = numerator / (denominator + epsilon)
    return np.mean(ratio) * 100