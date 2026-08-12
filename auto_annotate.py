import cv2
import numpy as np
import os
import glob
import tifffile as tiff

def main():
    # 1. Setup your paths
    IMAGE_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\HerbaceousVegetation"
    LABEL_DIR = r"D:\python\YOLO_VEGETATION_MAPPER\DATASET\labels"
    
    os.makedirs(LABEL_DIR, exist_ok=True)
    CLASS_ID = 0

    image_paths = []
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.PNG", "*.JPG", "*.JPEG", "*.TIF"]
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if len(image_paths) == 0:
        print(f"No images found in {IMAGE_DIR}")
        input("Press Enter to exit...")
        return

    print(f"Found {len(image_paths)} images. Starting NDVI auto-annotation...")
    success_count = 0

    for img_path in image_paths:
        mask = None
        h, w = 0, 0

        # Handle Multispectral TIFs
        if img_path.lower().endswith(('.tif', '.tiff')):
            try:
                tiff_img = tiff.imread(img_path)
                
                # Reshape if channels are first (e.g., 13, H, W)
                if len(tiff_img.shape) == 3 and tiff_img.shape[0] == 13:
                    tiff_img = np.transpose(tiff_img, (1, 2, 0))
                
                # If it's a 13-band Sentinel-2 image, calculate NDVI
                if len(tiff_img.shape) == 3 and tiff_img.shape[2] == 13:
                    h, w = tiff_img.shape[:2]
                    
                    # Band 4 (Index 3) is Red, Band 8 (Index 7) is NIR
                    red = tiff_img[:, :, 3].astype(np.float32)
                    nir = tiff_img[:, :, 7].astype(np.float32)
                    
                    # Calculate NDVI safely
                    denominator = (nir + red)
                    denominator[denominator == 0] = 1e-10 # Prevent division by zero
                    ndvi = (nir - red) / denominator
                    
                    # Create binary mask for vegetation (NDVI > 0.4 is a standard threshold)
                    # You can lower this to 0.2 if you want to capture sparser vegetation
                    mask = np.where(ndvi > 0.4, 255, 0).astype(np.uint8)
                else:
                    # Fallback for standard 3-band TIFs
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            except Exception as e:
                print(f"Skipping {os.path.basename(img_path)} due to read error.")
                continue

        # Handle standard JPG/PNGs
        else:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

        # Skip if mask generation failed
        if mask is None:
            continue

        # Find the outlines of the vegetation based on the generated mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_filepath = os.path.join(LABEL_DIR, label_filename)
        
        with open(label_filepath, "w") as f:
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    yolo_coords = []
                    for point in approx:
                        px, py = point[0]
                        nx = px / w
                        ny = py / h
                        yolo_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                    
                    if len(yolo_coords) >= 6:
                        f.write(f"{CLASS_ID} " + " ".join(yolo_coords) + "\n")
        
        success_count += 1
        if success_count % 100 == 0:
            print(f"Annotated {success_count} / {len(image_paths)} images...")

    print(f"\nDone! Automatically created YOLO labels for {success_count} images.")
    print(f"Check the folder: {LABEL_DIR}")
    input("Press Enter to exit...") 

if __name__ == '__main__':
    main()