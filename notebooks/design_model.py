import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 16), batch_size=None):
    if batch_size is not None:
        inputs = layers.Input(batch_shape=(batch_size,) + input_shape)
    else:
        inputs = layers.Input(shape=input_shape)
    
    # CNN Backbone using strided convolutions
    # Conv 1: 128x128x16 -> 64x64x4 (strides=2)
    x = layers.Conv2D(4, (5, 5), strides=(2, 2), padding='same', activation='relu')(inputs)
    
    # Conv 2: 64x64x4 -> 32x32x8 (strides=2)
    x = layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Conv 3: 32x32x8 -> 16x16x16 (strides=2)
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Conv 4: 16x16x16 -> 8x8x32 (strides=2)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Conv 5: 8x8x32 -> 4x4x32 (strides=2)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Flatten
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    
    # Dense layer
    x = layers.Dense(16, activation='relu')(x)
    
    # Single 3-unit output layer (linear, activations computed in custom loss / Rust)
    outputs = layers.Dense(3, name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="TinyNav_CustomCNN")
    return model

if __name__ == "__main__":
    model = build_model(batch_size=None)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n==========================================")
    print(f"Total Parameters: {total_params:,}")
    if total_params < 45000:
        print("STATUS: SUCCESS - Model is under the 45k limit!")
    else:
        print("STATUS: FAILED - Model is over the 45k limit!")
    print(f"==========================================\n")
    
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    print("Model compiled successfully!")
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model_baseline.keras")
    model.save(model_path)
    print(f"Baseline model structure saved to: {model_path}")
