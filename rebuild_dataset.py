import os
import glob
import random
import shutil
import cv2
import numpy as np
import tifffile as tiff

def main():
    print("Starting dataset rebuild...")
    
    # Paths
    ORIG_IMG_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\HerbaceousVegetation"
    ORIG_LBL_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\labels"
    BASE_OUT = r"D:\python\YOLO_VEGETATION_MAPPER\dataset_split"
    
    # 1. Wipe the old corrupted split folder and caches completely
    if os.path.exists(BASE_OUT):
        print("Clearing out old split folders and broken caches...")
        shutil.rmtree(BASE_OUT)
        
    for f in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(BASE_OUT, f), exist_ok=True)

    # 2. Get and shuffle all labels
    label_files = glob.glob(os.path.join(ORIG_LBL_DIR, "*.txt"))
    if not label_files:
        print("Error: No labels found in the original DATASET folder!")
        return
        
    random.shuffle(label_files)
    split_idx = int(len(label_files) * 0.8)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    def process_data(file_list, split_type):
        success = 0
        for txt_path in file_list:
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            matching_imgs = glob.glob(os.path.join(ORIG_IMG_DIR, base_name + ".*"))
            
            if not matching_imgs:
                continue
                
            img_path = matching_imgs[0]
            target_lbl = os.path.join(BASE_OUT, "labels", split_type, base_name + ".txt")
            target_img = os.path.join(BASE_OUT, "images", split_type, base_name + ".jpg") # FORCING JPG
            
            try:
                # 3. Extract RGB from 13-Band TIF and save as YOLO-friendly JPG
                if img_path.lower().endswith(('.tif', '.tiff')):
                    img = tiff.imread(img_path)
                    
                    # Fix channel order if necessary (13, H, W) -> (H, W, 13)
                    if len(img.shape) == 3 and img.shape[0] == 13: 
                        img = np.transpose(img, (1, 2, 0))
                        
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        if img.shape[2] == 13:
                            # Sentinel-2 RGB bands: B4(Red), B3(Green), B2(Blue)
                            r, g, b = img[:, :, 3], img[:, :, 2], img[:, :, 1]
                        else:
                            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                            
                        # Stack into BGR format for OpenCV
                        bgr = np.stack([b, g, r], axis=2).astype(np.float32)
                        
                        # Normalize pixel values to standard 0-255 image format
                        img_save = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imwrite(target_img, img_save)
                    else:
                        img_save = cv2.imread(img_path)
                        cv2.imwrite(target_img, img_save)
                else:
                    # Standard image fallback
                    img_save = cv2.imread(img_path)
                    cv2.imwrite(target_img, img_save)

                # 4. Copy the matching text file
                shutil.copy(txt_path, target_lbl)
                success += 1
                    
            except Exception as e:
                pass # Skip corrupted files silently
        
        print(f"Successfully processed {success} {split_type} pairs.")

    print("\nProcessing Training Data...")
    process_data(train_files, "train")
    
    print("\nProcessing Validation Data...")
    process_data(val_files, "val")

    # 5. Auto-generate the perfect dataset.yaml
    yaml_path = r"D:\python\YOLO_VEGETATION_MAPPER\dataset.yaml"
    with open(yaml_path, "w") as f:
        # Convert Windows backslashes to forward slashes for YOLO
        clean_path = BASE_OUT.replace("\\", "/")
        f.write(f"path: {clean_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: herbaceous_vegetation\n")
        
    print(f"\nRebuild Complete! dataset.yaml updated at {yaml_path}")

if __name__ == '__main__':
    main()