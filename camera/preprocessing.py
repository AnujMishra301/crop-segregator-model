"""
Real-Time Frame Preprocessing Module
Handles frame resizing, letterbox padding, color space conversion (BGR to RGB),
and normalization for neural network inference.
"""

import cv2
import numpy as np

class FramePreprocessor:
    """Configurable preprocessing unit converting raw camera frames to inference tensors."""

    def __init__(self, target_width=640, target_height=640, normalize=True):
        self.target_width = target_width
        self.target_height = target_height
        self.normalize = normalize

    def preprocess(self, frame_bgr):
        """Preprocesses raw BGR OpenCV camera frame.
        Returns:
            processed_tensor: NumPy array [1, 3, H, W] normalized float32 tensor
            scale_factor: tuple (scale_x, scale_y, pad_x, pad_y) for bounding box restoration
        """
        h_orig, w_orig = frame_bgr.shape[:2]
        
        # Calculate aspect ratio preserving scale
        scale = min(self.target_width / w_orig, self.target_height / h_orig)
        w_new = int(w_orig * scale)
        h_new = int(h_orig * scale)

        # Resize image
        resized = cv2.resize(frame_bgr, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # Create padded target canvas (letterboxing)
        padded_canvas = np.full((self.target_height, self.target_width, 3), 114, dtype=np.uint8)
        pad_x = (self.target_width - w_new) // 2
        pad_y = (self.target_height - h_new) // 2
        padded_canvas[pad_y:pad_y + h_new, pad_x:pad_x + w_new] = resized

        # BGR to RGB
        rgb = cv2.cvtColor(padded_canvas, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize [0, 1]
        tensor = rgb.astype(np.float32)
        if self.normalize:
            tensor /= 255.0

        # Transpose to [1, 3, H, W] for PyTorch / ONNX formats
        tensor_chw = np.transpose(tensor, (2, 0, 1))
        batch_tensor = np.expand_dims(tensor_chw, axis=0)

        scale_info = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "w_orig": w_orig,
            "h_orig": h_orig
        }

        return batch_tensor, scale_info, rgb

    def preprocess_tiles(self, frame_bgr, tile_size=400, overlap=50):
        """Generates overlapping 2D spatial tiles for detecting micro-weed seedlings (<32px).
        Returns list of (tile_tensor, (x_off, y_off, w_tile, h_tile)) tuples.
        """
        h_orig, w_orig = frame_bgr.shape[:2]
        tiles = []

        y = 0
        while y < h_orig:
            x = 0
            while x < w_orig:
                x_end = min(x + tile_size, w_orig)
                y_end = min(y + tile_size, h_orig)
                tile_crop = frame_bgr[y:y_end, x:x_end]

                # Preprocess individual tile
                tile_batch, tile_scale, _ = self.preprocess(tile_crop)
                tiles.append((tile_batch, (x, y, x_end - x, y_end - y)))

                if x_end == w_orig:
                    break
                x += (tile_size - overlap)

            if y_end == h_orig:
                break
            y += (tile_size - overlap)

        return tiles

if __name__ == "__main__":
    prep = FramePreprocessor(640, 640)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor, scale_info, _ = prep.preprocess(dummy_frame)
    tiles = prep.preprocess_tiles(dummy_frame)
    print(f"[Preprocessing Test] Tensor shape: {tensor.shape}, Generated {len(tiles)} high-res micro-weed tiles.")
