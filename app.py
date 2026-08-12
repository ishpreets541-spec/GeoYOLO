import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tifffile as tiff

# Configure the Streamlit page layout
st.set_page_config(page_title="Vegetation Cover Analysis", layout="wide")
st.title("Vegetation Detection and Cover Percentage")
st.write("Upload a satellite image to detect vegetation and calculate its fractional cover.")

# Load your custom trained model
@st.cache_resource
def load_model():
    model_path = r"D:\python\YOLO_VEGETATION_MAPPER\runs\segment\Vegetation_Project\yolo_veg_segmentation-4\weights\best.pt"
    return YOLO(model_path) 

model = load_model()

# Create a file uploader on the web page
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    try:
        # Check if the uploaded file is a Multispectral TIF
        if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
            # Read TIF directly from the uploaded memory stream
            img = tiff.imread(uploaded_file)
            
            # Handle channel-first format (13, H, W)
            if len(img.shape) == 3 and img.shape[0] == 13: 
                img = np.transpose(img, (1, 2, 0))
                
            # Extract RGB bands for visualization and YOLO
            if len(img.shape) == 3 and img.shape[2] >= 3:
                if img.shape[2] == 13:
                    # Sentinel-2 RGB bands: B4(Red), B3(Green), B2(Blue)
                    r, g, b = img[:, :, 3], img[:, :, 2], img[:, :, 1]
                else:
                    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                    
                # Stack into RGB format
                rgb = np.stack([r, g, b], axis=2).astype(np.float32)
                image_np = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                # Fallback for weirdly shaped TIFs
                image_np = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
        else:
            # Handle standard JPG/PNGs normally
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

        # Create two columns for a clean side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Image")
            # Streamlit displays the extracted numpy array directly
            st.image(image_np, use_container_width=True)
        
        # Create an analyze button
        if st.button("Analyze Vegetation", type="primary"):
            with st.spinner("Processing satellite imagery..."):
                
                # Run the AI prediction
                results = model.predict(image_np, retina_masks=True)
                result = results[0]
                
                with col2:
                    st.header("Analysis Results")
                    
                    # Check if the model actually found any vegetation masks
                    if result.masks is not None:
                        # Extract the mask data
                        masks = result.masks.data.cpu().numpy()
                        
                        # Merge all individual detection masks into one giant mask
                        combined_mask = np.any(masks, axis=0)
                        
                        # Calculate the percentage
                        total_pixels = combined_mask.size
                        vegetation_pixels = np.sum(combined_mask)
                        percentage = (vegetation_pixels / total_pixels) * 100
                        
                        # Generate the visual overlay with polygons
                        res_plotted = result.plot()
                        
                        # Convert BGR (OpenCV default) to RGB for the web
                        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        # Display the success metric and the mapped image
                        st.success(f"Calculated Vegetation Cover: {percentage:.2f}%")
                        st.image(res_plotted_rgb, use_container_width=True)
                    else:
                        st.warning("No vegetation detected in this image.")
                        st.image(image_np, use_container_width=True)
                        
    except Exception as e:
        st.error(f"Failed to process the image. Error details: {e}")