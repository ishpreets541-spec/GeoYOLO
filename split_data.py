import os
import random
import shutil
import glob

def main():
    # Source paths
    IMG_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\HerbaceousVegetation"
    LBL_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\labels"
    
    # New organized dataset folder
    BASE_OUT = r"D:\python\YOLO_VEGETATION_MAPPER\dataset_split"

    # Create the YOLO directory structure
    folders = ["images/train", "images/val", "labels/train", "labels/val"]
    for f in folders:
        os.makedirs(os.path.join(BASE_OUT, f), exist_ok=True)

    # Grab all generated text labels and shuffle them randomly
    label_files = glob.glob(os.path.join(LBL_DIR, "*.txt"))
    random.shuffle(label_files)

    # Calculate the 80/20 split
    split_idx = int(len(label_files) * 0.8)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    def copy_files(file_list, split_type):
        for txt_path in file_list:
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            
            # Find the matching image regardless of extension (.tif, .png, etc.)
            matching_imgs = glob.glob(os.path.join(IMG_DIR, base_name + ".*"))
            if not matching_imgs:
                continue
            img_path = matching_imgs[0]

            # Copy text and image to their new homes
            shutil.copy(txt_path, os.path.join(BASE_OUT, "labels", split_type, os.path.basename(txt_path)))
            shutil.copy(img_path, os.path.join(BASE_OUT, "images", split_type, os.path.basename(img_path)))

    print("Copying 80% to Training folders...")
    copy_files(train_files, "train")
    
    print("Copying 20% to Validation folders...")
    copy_files(val_files, "val")

    print(f"\nDone! Dataset successfully organized at: {BASE_OUT}")

if __name__ == '__main__':
    main()