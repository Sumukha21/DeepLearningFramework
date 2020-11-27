from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def resnet50_classifier(input_dim, no_classes, weights_path=None, verbose=False, weights_by_name=False, include_top=True,
                        last_layer_weights=True):
    init_model = ResNet50(input_shape=input_dim, weights=None, include_top=False)
    input_layer = init_model.layers[0].output
    x = init_model.layers[-1].output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if last_layer_weights:
        x = layers.Dense(no_classes, activation='softmax', name='probs')(x)
    else:
        x = layers.Dense(no_classes, activation='softmax', name='probs_no_weights')(x)
    model = Model(inputs=input_layer, outputs=x)
    if weights_path is not None:
        model.load_weights(weights_path, by_name=weights_by_name)
    if verbose:
        model.summary()
    return model


if __name__ == "__main1__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_dimension = [256, 256, 3]
    weights = "C:/Users/Sumukha/Desktop/Projects/Imagenet_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    load_weights_by_name = True
    resnet50_classifier(input_dimension, 2, weights_path=weights, weights_by_name=load_weights_by_name)
