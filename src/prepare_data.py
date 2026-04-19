from pathlib import Path #standard python library 
#Path allows us to handling the paths in this project with an OOP approach
import random #standard python library
import shutil #standard python library

random.seed(42) #set random for same result every run

RAW_DIR = Path("data/raw") #raw data directory initialized as a path object 
OUT_DIR = Path("data/processed") #processed data directory after this script initialized as a path object

TRAIN_RATIO = 0.70 #test, train, val split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_MAP = { # Map original classes to binary classes of malignant vs non-malignant where benign and normal map to non-malignant
    "malignant": "malignant",
    "benign": "non_malignant",
    "normal": "non_malignant",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"} #image file extension names

def is_real_image(path: Path) -> bool:
    """
    Logic: We only want real images (not masks) for our model. This function takes an image and returns whether it's a mask or not 
    Args: Image path
    Returns: Boolean (real image or not)
    Ex: Given "image.png" -> True
    """
    name = path.stem.lower() #The stem attribute of a path removes the extention (ex: my_file.txt -> my_file)
    return path.suffix.lower() in IMAGE_EXTS and not name.endswith("_mask") #The suffex attribute of a path is the file extension 

def split_files(files):
    return 0

def main():
    return 0

if __name__ == "__main__":
    main()