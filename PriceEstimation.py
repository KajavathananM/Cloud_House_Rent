import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # =========================================================================
    # STEP 1: Synthetic Dataset Generation & Storage
    # =========================================================================
    print("--- Step 1: Generating Dataset ---")
    np.random.seed(42)
    n_samples = 2000

    # Generate 6 housing attributes
    sqft = np.random.randint(400, 2500, n_samples)
    bedrooms = np.random.randint(1, 5, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    dist_city_center = np.round(np.random.uniform(0.5, 20.0, n_samples), 2)
    has_balcony = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    has_parking = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

    # Compute target rent_price with Gaussian noise
    rent_price = (
        150 + 
        (sqft * 0.8) + 
        (bedrooms * 120) + 
        (bathrooms * 90) - 
        (dist_city_center * 25) + 
        (has_balcony * 80) + 
        (has_parking * 110) + 
        np.random.normal(0, 40, n_samples)
    )

    df = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'dist_city_center': dist_city_center,
        'has_balcony': has_balcony,
        'has_parking': has_parking,
        'rent_price': np.round(rent_price, 2)
    })

    # Save reference dataset CSV
    df.to_csv("rental_housing_dataset.csv", index=False)


    # =========================================================================
    # STEP 2: Feature Engineering & Normalization Parameter Extraction
    # =========================================================================
    print("--- Step 2: Normalization & Feature Engineering ---")
    X = df.drop('rent_price', axis=1)
    y = df['rent_price']

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute mean (μ) and standard deviation (σ) for each feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Save mean and std to JSON for server/client inference consistency
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist()
    }
    
    with open("scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=4)
    print("Saved feature scaling parameters -> scaler_params.json")


    # =========================================================================
    # STEP 3: Neural Network Model Compilation & Training (MLP)
    # =========================================================================
    print("--- Step 3: Building & Training MLP Neural Network ---")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),           # Input Layer (6 features)
        tf.keras.layers.Dense(32, activation='relu'), # Hidden Layer 1 (32 neurons)
        tf.keras.layers.Dense(16, activation='relu'), # Hidden Layer 2 (16 neurons)
        tf.keras.layers.Dense(1)                      # Output Layer (1 continuous value)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train for 40 epochs with batch size 32
    model.fit(X_train_scaled, y_train, epochs=40, batch_size=32, verbose=0)


    # =========================================================================
    # STEP 4: Post-Training INT8 Quantization
    # =========================================================================
    print("--- Step 4: Applying INT8 Quantization ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset generator for calibrating INT8 ranges
    def representative_dataset():
        for i in range(100):
            yield [X_train_scaled[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()


    # =========================================================================
    # STEP 5: TFLite Asset Export
    # =========================================================================
    print("--- Step 5: Exporting TFLite Binary ---")
    with open("rental_valuation_quant.tflite", "wb") as f:
        f.write(tflite_model)
        
    print("Saved quantized model -> rental_valuation_quant.tflite")
    print(f"Model File Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    main()