import os
import cv2
import tensorflow as tf
import pandas as pd

class Data:
    def __init__(self, final_images_folder_path, final_data_csv_path) -> None:
        self.path = final_images_folder_path
        self.final_data = pd.read_csv(final_data_csv_path)
        self.data = []

    def make_training_data(self):
        for filename in os.listdir(self.path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_name_as_ts = os.path.splitext(filename)[0]
                data_row = self.final_data[self.final_data['timestamp'] == int(image_name_as_ts)]
                if not data_row.empty:
                    angle = data_row['angle'].iloc[0]
                    self.data.append((filename, angle))
        print(f"Total samples collected: {len(self.data)}")

    def load_image(self, filename):
        image = cv2.imread(os.path.join(self.path, filename))
        image = cv2.resize(image, (640, 480))  # Resize image to 640x480
        image = image / 255.0  # Normalize image data
        return image

    def data_generator(self, data):
        for filename, angle in data:
            image = self.load_image(filename)
            yield image, angle

    def create_dataset(self, data, batch_size, shuffle=True, repeat=True):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data),
            output_signature=(
                tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))  # Ensure buffer_size is appropriate
        if repeat:
            dataset = dataset.repeat()  # Repeat the dataset indefinitely
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def training_data(self, batch_size):
        data_train = self.data[:int((len(self.data) * 3) / 4)]
        print(f"Training samples: {len(data_train)}")
        return self.create_dataset(data_train, batch_size)

    def validation_data(self, batch_size):
        data_val = self.data[int((len(self.data) * 3) / 4):]
        print(f"Validation samples: {len(data_val)}")
        return self.create_dataset(data_val, batch_size, shuffle=False, repeat=True)  # Set repeat to True for validation as well

