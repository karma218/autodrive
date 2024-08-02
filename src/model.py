import tensorflow as tf
from tensorflow import keras
from keras import layers

@tf.keras.utils.register_keras_serializable()
def steering_angle_function(x):
    return tf.multiply(tf.math.atan(x), 2)

class SteeringModel:
    def __init__(self, width=640, height=480) -> None:
        self.width = width
        self.height = height
        self.model = self.build_model(self.width, self.height)

    def build_model(self, width, height):
        # Input layer
        inputs = keras.Input(name='input_shape', shape=(height, width, 3))

        # Convolutional feature maps
        x = layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(inputs)
        x = layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

        # Flatten layer
        x = layers.Flatten()(x)

        # Fully connected layers with dropouts for overfit protection
        x = layers.Dense(units=512, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=100, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=50, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=10, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)

        # Derive steering angle value from single output layer by point multiplication
        steering_angle = layers.Dense(units=1, activation='linear')(x)
        steering_angle = layers.Lambda(steering_angle_function, output_shape=(1,), name='steering_angle')(steering_angle)

        # Build and compile model
        model = keras.Model(inputs=[inputs], outputs=[steering_angle])
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={'steering_angle': 'mse'}
        )

        # Model summary
        model.summary()
        return model

    def train(self, name="default_model_name", data=None, epochs=20, steps=10, steps_val=10, batch_size=16):
        # Clear the TensorFlow session to free up memory
        tf.keras.backend.clear_session()

        # Extract training and validation datasets
        try:
            train_dataset = data.training_data(batch_size)
            val_dataset = data.validation_data(batch_size)
        except Exception as e:
            print(f"Error generating data for training! Reason: {e}")
            raise Exception(e)
        
        try:
            print("Starting Model Training...")
            # Define the ModelCheckpoint callback
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=f'{name}.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1,
            )
            
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Stop training after 5 epochs of no improvement
                verbose=1
            )
            # Train the model using the datasets
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                steps_per_epoch=steps,
                validation_steps=steps_val,
                callbacks=[checkpoint_callback, early_stopping_callback]
            )
            # Save the model
            print(f"Saving Model: {name}.keras")
            self.model.save(f'{name}.keras')
            return history
        except Exception as e:
            print(f"Error while training Model :( Reason : {e}")
            raise Exception(e)

    def verify_model(self, model_path, sample_image):
        try:
            # Load the saved model
            loaded_model = keras.models.load_model(model_path, custom_objects={"steering_angle_function": steering_angle_function})
            # Predict using the loaded model
            prediction = loaded_model.predict(tf.expand_dims(sample_image, axis=0))
            print(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            print(f"Error while verifying Model :( Reason : {e}")
            raise Exception(e)
