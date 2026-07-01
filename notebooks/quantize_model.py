import os
import sys
import numpy as np
import tensorflow as tf

# Add notebooks directory to path to import design_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from design_model import build_model

BEST_MODEL_FILE = "models/best_model.keras"
DATASET_FILE = "dataset/preprocessed_data.npz"
TFLITE_OUT_FILE = "models/tiny_nav_quantized.tflite"
RUST_OUT_FILE = "esp32-rust/src/model_data.rs"
C_OUT_FILE = "models/model_data.h"

def representative_data_gen():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file {DATASET_FILE} not found. Run preprocess_dataset.py first.")
    
    data = np.load(DATASET_FILE)
    x_train = data["x_train"]
    
    for i in range(len(x_train)):
        sample = x_train[i:i+1].astype(np.float32) / 255.0
        yield [sample]

def main():
    print(f"Loading trained weights from {BEST_MODEL_FILE}...")
    if not os.path.exists(BEST_MODEL_FILE):
        print(f"Error: Trained model {BEST_MODEL_FILE} not found. Run train_model.py first.")
        return
        
    # Load dynamic model trained on batches (need to pass custom objects)
    from train_model import custom_loss, steer_mae, throt_mae, obst_accuracy
    model_dynamic = tf.keras.models.load_model(
        BEST_MODEL_FILE,
        custom_objects={
            'custom_loss': custom_loss,
            'steer_mae': steer_mae,
            'throt_mae': throt_mae,
            'obst_accuracy': obst_accuracy
        }
    )
    
    # Build a static model with fixed batch size = 1 for deployment
    print("Building static model for deployment (batch_size=1)...")
    model_static = build_model(batch_size=1)
    
    # Copy weights from dynamic to static model
    print("Transferring weights to static model...")
    model_static.set_weights(model_dynamic.get_weights())
    
    # Initialize TFLite Converter on the static model
    print("Configuring TensorFlow Lite Converter for Post-Training INT8 Quantization (Static Linear Graph)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_static)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    
    # Enforce full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Converting model...")
    try:
        tflite_model = converter.convert()
        print("Model conversion successful!")
    except Exception as e:
        print(f"Error during quantization: {e}")
        return
        
    # Save the TFLite model
    os.makedirs(os.path.dirname(TFLITE_OUT_FILE), exist_ok=True)
    with open(TFLITE_OUT_FILE, "wb") as f:
        f.write(tflite_model)
    print(f"Saved quantized model to: {TFLITE_OUT_FILE} ({len(tflite_model) / 1024:.2f} KB)")
    
    # Export as C/C++ Header file
    print(f"Exporting model bytes to C header: {C_OUT_FILE}...")
    hex_bytes_c = [f"0x{b:02x}" for b in tflite_model]
    c_content = f"""// Auto-generated TinyNav quantized model bytes
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char model_data[] = {{
    {', '.join(hex_bytes_c)}
}};
const unsigned int model_data_len = {len(tflite_model)};

#endif // MODEL_DATA_H
"""
    with open(C_OUT_FILE, "w") as f:
        f.write(c_content)
    print("C header exported successfully.")
    
    # Export as Rust source file
    print(f"Exporting model bytes to Rust slice: {RUST_OUT_FILE}...")
    hex_bytes_rs = [f"0x{b:02x}" for b in tflite_model]
    
    os.makedirs(os.path.dirname(RUST_OUT_FILE), exist_ok=True)
    rust_content = f"""// Auto-generated TinyNav quantized model bytes for Rust deployment
pub static MODEL_DATA: &[u8] = &[
    {', '.join(hex_bytes_rs)}
];
"""
    with open(RUST_OUT_FILE, "w") as f:
        f.write(rust_content)
    print("Rust source file exported successfully.")
    
    print("\n==========================================")
    print("Quantization and export completed successfully!")
    print("==========================================\n")

if __name__ == "__main__":
    main()
