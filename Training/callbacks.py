import os
from tensorflow.keras.callbacks import ModelCheckpoint


def model_checkpoint_initializer(save_folder, save_path, save_best_only=True):
    save_name = os.path.join(save_folder, save_path)
    return ModelCheckpoint(filepath=save_name, save_best_only=save_best_only, save_freq="epoch")
