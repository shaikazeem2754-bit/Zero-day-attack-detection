import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class ContrastiveAutoencoder(keras.Model):
    """Contrastive Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim, latent_dim=32, name="contrastive_autoencoder"):
        super(ContrastiveAutoencoder, self).__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_threshold = None
        
        # Encoder architecture
        self.encoder = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(latent_dim, activation='relu', name='latent_space')
        ], name='encoder')
        
        # Decoder architecture
        self.decoder = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='sigmoid', name='reconstruction')
        ], name='decoder')
        
    def call(self, x, training=None):
        """Forward pass"""
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def get_latent_features(self, x):
        """Get latent space representations"""
        return self.encoder(x, training=False)
    
    def compute_reconstruction_error(self, x):
        """Compute reconstruction error for input samples"""
        reconstructions = self(x, training=False)
        mse = tf.reduce_mean(tf.square(x - reconstructions), axis=1)
        return mse.numpy()
    
    def fit_with_threshold(self, X_train, X_val=None, epochs=100, batch_size=32, validation_split=0.2):
        """Train autoencoder and compute reconstruction threshold"""
        # Compile model
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse')
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        if X_val is None:
            history = self.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            history = self.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
        
        # Compute reconstruction errors for training data
        train_reconstructions = self(X_train, training=False)
        train_errors = tf.reduce_mean(tf.square(X_train - train_reconstructions), axis=1)
        
        # Set threshold (mean + 2*std for normal data)
        self.reconstruction_threshold = np.mean(train_errors) + 2 * np.std(train_errors)
        
        return history
    
    def detect_anomalies(self, X):
        """Detect anomalies based on reconstruction error"""
        if self.reconstruction_threshold is None:
            raise ValueError("Model not trained. Call fit_with_threshold first.")
        
        errors = self.compute_reconstruction_error(X)
        is_anomaly = errors > self.reconstruction_threshold
        
        # Calculate anomaly scores (0-1 range)
        max_error = np.max(errors)
        min_error = np.min(errors)
        if max_error > min_error:
            anomaly_scores = (errors - min_error) / (max_error - min_error)
        else:
            anomaly_scores = np.zeros_like(errors)
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_errors': errors,
            'anomaly_scores': anomaly_scores,
            'threshold': self.reconstruction_threshold
        }