import os
import glob
import cv2
import numpy as np
import tifffile as tiff

def main():
    # Target the split dataset folders
    base_dir = r"D:\python\YOLO_VEGETATION_MAPPER\dataset_split\images"
    
    # Find all TIF files in train and val folders
    tif_files = glob.glob(os.path.join(base_dir, "**", "*.tif"), recursive=True)
    tif_files.extend(glob.glob(os.path.join(base_dir, "**", "*.tiff"), recursive=True))
    
    if not tif_files:
        print("No TIF files found. If your folder only has PNGs/JPGs, the issue is your dataset.yaml.")
        return

    print(f"Found {len(tif_files)} TIF images. Converting to YOLO-friendly RGB JPGs...")
    
    success_count = 0
    for img_path in tif_files:
        try:
            img = tiff.imread(img_path)
            
            # Handle channel-first format (13, H, W)
            if len(img.shape) == 3 and img.shape[0] == 13:
                img = np.transpose(img, (1, 2, 0))
                
            # Extract RGB for YOLO to learn from visually
            if len(img.shape) == 3 and img.shape[2] >= 3:
                if img.shape[2] == 13:
                    # Sentinel-2 RGB: Band 4 (Red), Band 3 (Green), Band 2 (Blue)
                    r = img[:, :, 3].astype(np.float32)
                    g = img[:, :, 2].astype(np.float32)
                    b = img[:, :, 1].astype(np.float32)
                else:
                    # Generic fallback: just take the first 3 bands
                    r = img[:, :, 0].astype(np.float32)
                    g = img[:, :, 1].astype(np.float32)
                    b = img[:, :, 2].astype(np.float32)
                    
                # Stack into BGR format (OpenCV's default color order)
                bgr = np.stack([b, g, r], axis=2)
                
                # Normalize pixel values to 0-255 for standard image format
                bgr_norm = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Save as JPG
                new_path = os.path.splitext(img_path)[0] + ".jpg"
                cv2.imwrite(new_path, bgr_norm)
                
                # Delete the original TIF so YOLO doesn't get confused
                os.remove(img_path)
                success_count += 1
                
        except Exception as e:
            print(f"Skipped {os.path.basename(img_path)}: {e}")
            
    print(f"\nSuccessfully converted {success_count} images to JPG format!")

if __name__ == '__main__':
    main()