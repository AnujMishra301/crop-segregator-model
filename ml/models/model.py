"""
Object Detection Model Architecture Definition
Wraps Ultralytics YOLO Nano (YOLOv8n / YOLOv11n) with PyTorch Torchvision fallback.
Outputs detections with format: (class_id, class_name, confidence, x_min, y_min, x_max, y_max).
"""

import os
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class BaselineDetector(nn.Module):
    """Wrapper class for baseline object detection model."""
    def __init__(self, num_classes=4, architecture="yolov8n", pretrained_weights="yolov8n.pt"):
        super(BaselineDetector, self).__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        self.ultralytics_model = None

        if ULTRALYTICS_AVAILABLE:
            print(f"[Model Init] Loading Ultralytics {architecture} (num_classes={num_classes})...")
            self.ultralytics_model = YOLO(architecture)
        else:
            print(f"[Model Init] Ultralytics not found. Fallback PyTorch MobileNet / SSDLite initialized.")
            # PyTorch Torchvision SSDLite MobileNetV3 fallback
            import torchvision
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large
            self.pytorch_model = ssdlite320_mobilenet_v3_large(num_classes=num_classes + 1)

    def forward(self, x):
        if self.ultralytics_model is not None:
            return self.ultralytics_model(x)
        else:
            return self.pytorch_model(x)

def get_model(config):
    """Factory function creating baseline detector based on YAML configuration."""
    num_classes = config.get("num_classes", 4)
    architecture = config.get("architecture", "yolov8n")
    weights = config.get("weights_pretrained", "yolov8n.pt")
    
    if ULTRALYTICS_AVAILABLE:
        model = YOLO(weights if os.path.exists(weights) else architecture)
        return model
    else:
        model = BaselineDetector(num_classes=num_classes, architecture=architecture)
        return model

if __name__ == "__main__":
    from ml.config.config_loader import load_config
    cfg = load_config()
    m = get_model(cfg)
    print("Baseline Model Initialized Successfully.")
