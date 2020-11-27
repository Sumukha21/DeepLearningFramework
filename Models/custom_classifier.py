from tensorflow.keras import layers
from tensorflow.keras.models import Model


def feature_extraction_block(n_filters, x):
    x = layers.Conv2D(n_filters, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling2D()(x)
    return x


def custom_classifier(input_dim, no_classes, final_activation="softmax", weights=None, last_layer_weights=True,
                      weights_by_name=False, verbose=False, feature_extraction_filters=(32, 64, 128, 256)):

    input_layer = layers.Input(input_dim)
    x = input_layer
    for filter_no in feature_extraction_filters:
        x = feature_extraction_block(filter_no, x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.Dropout(0.5)(x)

    if last_layer_weights:
        x = layers.Dense(no_classes, activation=final_activation)(x)
    else:
        x = layers.Dense(no_classes, activation=final_activation, name="no_weights")(x)

    model = Model(inputs=input_layer, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=weights_by_name)

    if verbose:
        model.summary()

    return model


if __name__ == "__main1__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_dimension = [256, 256, 3]
    classes = 2
    model1 = custom_classifier(input_dimension, classes, verbose=True)