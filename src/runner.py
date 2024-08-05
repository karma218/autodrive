from data import Data
from model import SteeringModel
import os
import sys
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import config as keras_config  # Importing keras config

def run_training():
    curr_dir = os.getcwd()
    dataset_folder = os.path.join(curr_dir, "../dataset")
    image_path = os.path.join(dataset_folder, "final_images_center")
    csv_path = os.path.join(dataset_folder, "final_data_center.csv")

    data = Data(image_path, csv_path)
    
    try:
        data.make_training_data()
    except Exception as e:
        print(f"Error making Training Data! {e}")
        return

    try:
        model = SteeringModel(640, 480)
    except Exception as e:
        print(f"Error building model :( Reason: {e}")
        return

    try:
        name = "test"
        batch_size = 64  # Reduced batch size
        epochs = 50
        steps_per_epoch = 30
        validation_steps = 15

        history = model.train(name=name, data=data, epochs=epochs, steps=steps_per_epoch, steps_val=validation_steps, batch_size=batch_size)
        print(history)
    except Exception as e:
        print(f"Error Training Model :( Reason: {e}")
        raise Exception(e)

def verify_model(model_path, sample_image_path):
    try:
        # Configure TensorFlow to use fewer threads
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Enable unsafe deserialization for Keras Lambda layers
        keras_config.enable_unsafe_deserialization()

        # Load the saved model
        loaded_model = keras.models.load_model(model_path, custom_objects={"tf": tf})

        # Read the sample image
        sample_image = cv2.imread(sample_image_path)
        if sample_image is None:
            raise ValueError(f"Error: Unable to read the image file at {sample_image_path}. Please check the file path.")

        # Resize the sample image
        sample_image = cv2.resize(sample_image, (640, 480))
        sample_image = sample_image / 255.0  # Normalize image data

        # Predict using the loaded model
        prediction = loaded_model.predict(tf.expand_dims(sample_image, axis=0))
        print(f"Sample Prediction: {prediction}")

    except Exception as e:
        print(f"Error verifying model :( Reason: {e}")
        raise Exception(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py [train|verify] [model_path] [sample_image_path]")
    else:
        if sys.argv[1] == "train":
            run_training()
        elif sys.argv[1] == "verify" and len(sys.argv) == 4:
            verify_model(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python runner.py [train|verify] [model_path] [sample_image_path]")
