import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU training locally
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_FILE = "dataset/preprocessed_data.npz"
BASE_MODEL_FILE = "models/model_baseline.keras"
BEST_MODEL_FILE = "models/best_model.keras"

BATCH_SIZE = 16
EPOCHS = 60  # Increased for training with larger dataset and heavy augmentation

def data_generator(X, Y_steer, Y_throt, Y_obst, batch_size=8, augment=True):
    num_samples = len(X)
    while True:
        indices = np.arange(num_samples)
        if augment:
            np.random.shuffle(indices)
            
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            # Read uint8 data and convert to float32 [0.0, 1.0]
            x_batch = X[batch_indices].astype(np.float32) / 255.0
            y_s_batch = Y_steer[batch_indices].copy()
            y_t_batch = Y_throt[batch_indices].copy()
            y_o_batch = Y_obst[batch_indices].copy()
            
            if augment:
                for i in range(len(x_batch)):
                    # 1. Horizontal Flip (50% chance)
                    if np.random.rand() > 0.5:
                        x_batch[i] = np.flip(x_batch[i], axis=1)  # Flip along width axis (axis 1)
                        y_s_batch[i] = -y_s_batch[i]              # Invert steering label
                        
                    # 2. Horizontal Shift (Translation) with steering compensation
                    dx = np.random.randint(-12, 13) # Shift up to 12 pixels left or right
                    if dx > 0:
                        # Shift right: pad left with edge values
                        x_batch[i][:, dx:, :] = x_batch[i][:, :-dx, :]
                        x_batch[i][:, :dx, :] = x_batch[i][:, dx:dx+1, :]
                    elif dx < 0:
                        # Shift left: pad right with edge values
                        x_batch[i][:, :dx, :] = x_batch[i][:, -dx:, :]
                        x_batch[i][:, dx:, :] = x_batch[i][:, dx-1:dx, :]
                    # Compensate steering: shifting image right means track is to the right, steer right
                    y_s_batch[i] = np.clip(y_s_batch[i] + dx * 0.05, -1.0, 1.0)

                    # 3. Brightness Adjustment (consistent across all channels in stack)
                    brightness_factor = np.random.uniform(0.8, 1.2)
                    x_batch[i] = np.clip(x_batch[i] * brightness_factor, 0.0, 1.0)

                    # 4. Contrast Adjustment
                    contrast_factor = np.random.uniform(0.8, 1.2)
                    mean_val = np.mean(x_batch[i])
                    x_batch[i] = np.clip((x_batch[i] - mean_val) * contrast_factor + mean_val, 0.0, 1.0)

                    # 5. Gaussian Noise (simulating camera sensor noise)
                    if np.random.rand() > 0.5:
                        noise = np.random.normal(0, np.random.uniform(0.01, 0.04), x_batch[i].shape)
                        x_batch[i] = np.clip(x_batch[i] + noise, 0.0, 1.0)

                    # 6. Motion Blur (randomly blur the image)
                    if np.random.rand() > 0.7:
                        x_batch[i] = (x_batch[i] + np.roll(x_batch[i], 1, axis=0) + np.roll(x_batch[i], -1, axis=0) + np.roll(x_batch[i], 1, axis=1) + np.roll(x_batch[i], -1, axis=1)) / 5.0
            
            # Stack outputs: (batch_size, 3)
            y_batch = np.column_stack([y_s_batch, y_t_batch, y_o_batch])
            yield x_batch, y_batch

def custom_loss(y_true, y_pred):
    # Split predictions and ground truths
    y_true_steer = y_true[:, 0:1]
    y_true_throt = y_true[:, 1:2]
    y_true_obst = y_true[:, 2:3]
    
    y_pred_steer = y_pred[:, 0:1]
    y_pred_throt = y_pred[:, 1:2]
    y_pred_obst = y_pred[:, 2:3]
    
    # Apply activations internally
    steer_pred_act = tf.tanh(y_pred_steer)
    throt_pred_act = tf.sigmoid(y_pred_throt)
    obst_pred_act = tf.sigmoid(y_pred_obst)
    
    # Calculate individual losses
    steer_loss = tf.reduce_mean(tf.square(y_true_steer - steer_pred_act))
    throt_loss = tf.reduce_mean(tf.square(y_true_throt - throt_pred_act))
    obst_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_obst, obst_pred_act))
    
    # Combine losses with weights: w_steer = 1.0, w_throt = 1.0, w_obst = 0.5
    total_loss = steer_loss * 1.0 + throt_loss * 1.0 + obst_loss * 0.5
    return total_loss

def steer_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0:1] - tf.tanh(y_pred[:, 0:1])))

def throt_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1:2] - tf.sigmoid(y_pred[:, 1:2])))

def obst_accuracy(y_true, y_pred):
    pred_rounded = tf.round(tf.sigmoid(y_pred[:, 2:3]))
    return tf.reduce_mean(tf.cast(tf.equal(y_true[:, 2:3], pred_rounded), tf.float32))

def main():
    # 1. Load preprocessed data
    print(f"Loading preprocessed data from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print(f"Error: Preprocessed data file {DATASET_FILE} does not exist. Run preprocess_dataset.py first.")
        return
        
    data = np.load(DATASET_FILE)
    x_train, y_tr_steer, y_tr_throt, y_tr_obst = data["x_train"], data["y_train_steer"], data["y_train_throt"], data["y_train_obst"]
    x_val, y_val_steer, y_val_throt, y_val_obst = data["x_val"], data["y_val_steer"], data["y_val_throt"], data["y_val_obst"]
    x_test, y_te_steer, y_te_throt, y_te_obst = data["x_test"], data["y_test_steer"], data["y_test_throt"], data["y_test_obst"]
    
    print(f"Loaded {len(x_train)} train, {len(x_val)} val, {len(x_test)} test samples.")
    
    # 2. Load model (resume from best_model if it exists, otherwise load baseline)
    if os.path.exists(BEST_MODEL_FILE):
        print(f"Found existing trained model at {BEST_MODEL_FILE}. Resuming training from existing weights...", flush=True)
        model = tf.keras.models.load_model(
            BEST_MODEL_FILE,
            custom_objects={
                "custom_loss": custom_loss,
                "steer_mae": steer_mae,
                "throt_mae": throt_mae,
                "obst_accuracy": obst_accuracy
            }
        )
    else:
        print(f"Loading baseline model from {BASE_MODEL_FILE}...", flush=True)
        if not os.path.exists(BASE_MODEL_FILE):
            print("Error: Baseline model file does not exist. Run design_model.py first.", flush=True)
            return
        model = tf.keras.models.load_model(BASE_MODEL_FILE)
    
    # 3. Re-compile model with custom single-head loss and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=custom_loss,
        metrics=[steer_mae, throt_mae, obst_accuracy]
    )
    print("Model compiled with custom loss and metrics successfully.")
    
    # 4. Setup generators
    train_gen = data_generator(x_train, y_tr_steer, y_tr_throt, y_tr_obst, batch_size=BATCH_SIZE, augment=True)
    val_gen = data_generator(x_val, y_val_steer, y_val_throt, y_val_obst, batch_size=BATCH_SIZE, augment=False)
    
    # Calculate steps per epoch
    train_steps = int(np.ceil(len(x_train) / BATCH_SIZE))
    val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
    
    # 5. Define Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_FILE,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 6. Run training loop
    print(f"Starting model training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training finished.")
    
    # 7. Evaluate on Test set
    print("Evaluating model on Test set...")
    x_test_normalized = x_test.astype(np.float32) / 255.0
    y_test_stacked = np.column_stack([y_te_steer, y_te_throt, y_te_obst])
    results = model.evaluate(
        x_test_normalized,
        y_test_stacked,
        verbose=0
    )
    
    # Print individual metric results
    metric_names = model.metrics_names
    print("\n=== TEST METRICS RESULTS ===")
    for name, val in zip(metric_names, results):
        print(f"Test {name}: {val:.4f}")
    print("============================\n")

if __name__ == "__main__":
    main()
