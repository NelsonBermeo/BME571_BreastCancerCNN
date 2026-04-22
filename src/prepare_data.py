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

#Ensuring the directories is not needed

def collect_files():
    grouped = {"malignant": [], "non_malignant": []} #This will hold all the file paths 

    for raw_class, target_class in CLASS_MAP.items(): #Loop through the class_map above to line 39 -> create path dir / class and 
        class_dir = RAW_DIR / raw_class
        if not class_dir.exists(): #Quick check to make sure dir works 
            print(f"Error: missing folder {class_dir}")
            continue

        files = [p for p in class_dir.iterdir() if p.is_file() and is_real_image(p)] #iterdir() loops through the folder and is_file() checks we are in not in a subfolder and is_real_image is a function we defined before to ignore masks 
        #ex: 
        #From this: 
        # data/raw/benign/
        # image1.png
        # image1_mask.png
        # image2.png
        # We get -> files = [image1.png, image2.png]
        grouped[target_class].extend(files) 
        # Group ends up being for example: 
        # grouped = {
        # "malignant": [
        #     path/to/malignant1.png,
        #     path/to/malignant2.png
        # ],
        # "non_malignant": [
        #     path/to/benign1.png,
        #     path/to/normal1.png,
        #     path/to/benign2.png
        # ]
        # }
    return grouped

def split_files(files):
    random.shuffle(files) #Shuffling the data so we can just have it randomly
    n = len(files) #Get the length (number of files)
    n_train = int(n * TRAIN_RATIO) #(number of train files)
    n_val = int(n * VAL_RATIO) #(number of val files)

    train_files = files[:n_train] #Slice to first train number of files
    val_files = files[n_train:n_train + n_val] #Slice next number of files for val
    test_files = files[n_train + n_val:]#Slice remainder number of files 
    return train_files, val_files, test_files #return those subsets in 3 lists

def copy_split(files, split, cls):
    #Up to this point we only had lists of paths, this function will copy the image into the right folder based on our previous splits
    for file_path in files: 
        dest = OUT_DIR / split / cls / file_path.name
        shutil.copy2(file_path, dest) #Refered to https://docs.python.org/3/library/shutil.html
    #Ex: 
    #We copy from data/raw/benign/image1.png -> data/processed/train/non_malignant/image1.png

def main():
    #Using all the functions from previous to split the data into their correct folders 
    grouped = collect_files() #This from earlier collects the images into two bunches 

    #We loop through the files we just defined using the items() method which splits keys and values to use within the loop
    for malignancy, files in grouped.items():
        train_files, val_files, test_files = split_files(files) #Split using the earlier function

        copy_split(train_files, "train", malignancy) #Copy them into the proper areas
        copy_split(val_files, "val", malignancy)
        copy_split(test_files, "test", malignancy)

        print(f"{malignancy}:")
        print(f"  train = {len(train_files)}")
        print(f"  val   = {len(val_files)}")
        print(f"  test  = {len(test_files)}")

if __name__ == "__main__":
    main()

#Upon running, the files were copied correctly and the output was: 
# malignant:
#   train = 147
#   val   = 31
#   test  = 33
# non_malignant:
#   train = 410
#   val   = 88
#   test  = 89