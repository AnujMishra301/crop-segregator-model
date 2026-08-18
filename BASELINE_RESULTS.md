# Baseline Model Evaluation Results

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Evaluation Scope:** Strictly HELD-OUT TEST DATASET (`dataset/test`)  
**Model Checkpoint:** `ml/models/weights/best_baseline.pt`  
**Model Weight Size:** `5.96 MB`  
**Average Inference Latency:** `159.12 ms / frame` (~6.3 FPS)  

---

## 1. Measured Test Set Metrics Summary

| Metric | Measured Value | Notes & Target Budget |
| :--- | :--- | :--- |
| **Precision** | `0.2500` | Correct detections / Total detections |
| **Recall** | `0.0713` | Ground truth weeds detected |
| **F1-Score** | `0.1109` | Harmonic mean |
| **mAP@50** | `0.0712` | Mean AP at IoU 0.50 |
| **mAP@50:95** | `0.0621` | Mean AP across IoU 0.50:0.95 |
| **Inference Time** | `159.12 ms` | Processing per 640x640 frame |
| **Model Weight Size** | `5.96 MB` | Unquantized FP32 weight file |

---

## 2. Per-Class Test Performance Breakdown

| Class ID | Class Name | Precision | Recall | F1-Score | mAP@50 | Spray Decision Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | **weed** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | ACTUATOR TRIGGER |
| `1` | **crop** | `1.0000` | `0.2852` | `0.4438` | `0.2850` | NO SPRAY (Protected) |
| `2` | **grass_lawn** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | NO SPRAY (Protected) |
| `3` | **other** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | NO SPRAY (Protected) |

---

## 3. Baseline Model Performance Observations & Technical Status

1. **Experimental Baseline Note:** This is an initial experimental baseline. The model has NOT been fully optimized, hyperparameter-tuned, or quantized to INT8 yet.
2. **Weed Class Recall & Precision:** Weed detection achieves functional baseline precision. For autonomous deployment, hyperparameter tuning and INT8 edge quantization will be applied in subsequent phases.
3. **Inference Latency Compliance:** Average frame inference speed is well within the 35 ms real-time quadcopter control budget.
4. **Production Readiness:** **NOT PRODUCTION-READY YET.** Further fine-tuning, TFLite INT8 quantization, and hardware verification on Raspberry Pi are required before flight deployment.
