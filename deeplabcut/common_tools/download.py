from huggingface_hub import hf_hub_download
import os

def download_from_huggingface_hub(target_folder_path, repo_id, filename, subfolder=None):
    """
    Download a file from the Hugging Face Hub to a specified local directory.
    
    Parameters:
        target_folder_path (str): Local directory path where file will be saved
        repo_id (str): Hugging Face repository ID (e.g., 'noahcao/sapiens-pose-coco')
        filename (str): Name of the file to download
        subfolder (str, optional): Path to subfolder within the repository where the file is located
        
    Returns:
        str: Full path to the downloaded file
    
    Examples:
        >>> # Download a model file from noahcao/sapiens-pose-coco
        >>> download_from_huggingface_hub(
        ...     target_folder_path="./models/sapiens",
        ...     repo_id="noahcao/sapiens-pose-coco",
        ...     filename="sapiens_2b_coco_best_coco_AP_822_torchscript.pt2",
        ...     subfolder="sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b"
        ... )
        
        >>> # Download a file without specifying a subfolder
        >>> download_from_huggingface_hub(
        ...     target_folder_path="./data",
        ...     repo_id="noahcao/sapiens-pose-coco",
        ...     filename="COCO_val2017_detections_AP_H_70_person.json",
        ...     subfolder="sapiens_host/pose/person_detection_results"
        ... )
    """
    # Create the target directory if it does not exist
    os.makedirs(target_folder_path, exist_ok=True)

    # Download the file from Hugging Face Hub
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=target_folder_path
    )
    
# Example usage:
if __name__ == "__main__":
    # Example to download the model from noahcao/sapiens-pose-coco repository
    target_dir = "xxxx/sapiens/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b"
    
    downloaded_file = download_from_huggingface_hub(
        target_folder_path=target_dir,
        repo_id="noahcao/sapiens-pose-coco",
        filename="sapiens_2b_coco_best_coco_AP_822_torchscript.pt2",
        subfolder="sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b"
    )
    
    print(f"File downloaded to: {downloaded_file}")
    