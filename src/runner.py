from data import Data
from model import SteeringModel
import os

def run():
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
        name = input("Provide model name: ")
        batch_size = 4  # Reduced batch size
        epochs = 30
        steps_per_epoch = 15
        validation_steps = 15

        history = model.train(name=name, data=data, epochs=epochs, steps=steps_per_epoch, steps_val=validation_steps, batch_size=batch_size)
        print(history)
    except Exception as e:
        print(f"Error Training Model :( Reason: {e}")
        raise Exception(e)

if __name__ == "__main__":
    run()
