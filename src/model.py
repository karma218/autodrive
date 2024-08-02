import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt

@tf.keras.utils.register_keras_serializable()
def steering_angle_function(x):
    return tf.multiply(tf.math.atan(x), 2)

class SteeringModel:
    def __init__(self, width=640, height=480) -> None:
        self.width = width
        self.height = height

    def build_model(self, hp):
        # Input layer
        inputs = keras.Input(name='input_shape', shape=(self.height, self.width, 3))

        # Convolutional feature maps
        x = layers.Conv2D(filters=hp.Int('filters_1', min_value=24, max_value=64, step=8),
                          kernel_size=(5, 5), strides=(2, 2), activation='relu')(inputs)
        x = layers.Conv2D(filters=hp.Int('filters_2', min_value=24, max_value=64, step=8),
                          kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(filters=hp.Int('filters_3', min_value=24, max_value=64, step=8),
                          kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(filters=hp.Int('filters_4', min_value=24, max_value=64, step=8),
                          kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2D(filters=hp.Int('filters_5', min_value=24, max_value=64, step=8),
                          kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

        # Flatten layer
        x = layers.Flatten()(x)

        # Fully connected layers with dropouts for overfit protection
        x = layers.Dense(units=hp.Int('units_1', min_value=128, max_value=512, step=128), activation='relu')(x)
        x = layers.Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = layers.Dense(units=hp.Int('units_2', min_value=64, max_value=256, step=64), activation='relu')(x)
        x = layers.Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = layers.Dense(units=hp.Int('units_3', min_value=32, max_value=128, step=32), activation='relu')(x)
        x = layers.Dropout(rate=hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = layers.Dense(units=hp.Int('units_4', min_value=8, max_value=32, step=8), activation='relu')(x)
        x = layers.Dropout(rate=hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1))(x)

        # Derive steering angle value from single output layer by point multiplication
        steering_angle = layers.Dense(units=1, activation='linear')(x)
        steering_angle = layers.Lambda(steering_angle_function, output_shape=(1,), name='steering_angle')(steering_angle)

        # Build and compile model
        model = keras.Model(inputs=[inputs], outputs=[steering_angle])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss={'steering_angle': 'mse'}
        )

        return model

    def train(self, name="default_model_name", data=None, epochs=50, steps=51, steps_val=17, batch_size=32):
        # Clear the TensorFlow session to free up memory
        tf.keras.backend.clear_session()

        # Extract training and validation datasets
        try:
            train_dataset = data.training_data(batch_size)
            val_dataset = data.validation_data(batch_size)
        except Exception as e:
            print(f"Error generating data for training! Reason: {e}")
            raise Exception(e)

        tuner = kt.Hyperband(
            self.build_model,
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='my_dir',
            project_name='steering_model_tuning'
        )

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units_1')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)

        # Build the model with the optimal hyperparameters and train it
        model = tuner.hypermodel.build(best_hps)
        
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

        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, steps_per_epoch=steps, validation_steps=steps_val, callbacks=[checkpoint_callback, early_stopping_callback])

        # Save the model
        print(f"Saving Model: {name}.keras")
        model.save(f'{name}.keras')
        return history

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
