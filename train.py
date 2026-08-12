from ultralytics import YOLO
import torch

def main():
    # Automatically use your GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on: {device}")

    # Load the YOLOv8 nano segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train!
    results = model.train(
        data='dataset.yaml',          
        epochs=10,                    # Start with 10 passes through your 6,000 images
        imgsz=640,                    
        batch=16,                     
        project='Vegetation_Project', 
        name='yolo_veg_segmentation', 
        device=device,
        patience=10                   # Stop early if accuracy stops improving
    )
    print("Training complete! Weights saved in Vegetation_Project/yolo_veg_segmentation/weights/best.pt")

if __name__ == '__main__':
    main()