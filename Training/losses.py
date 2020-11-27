from tensorflow.keras.losses import categorical_crossentropy


def categorical_cross_entropy_loss(dummy=1e-6):
    def loss(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)
    return loss
