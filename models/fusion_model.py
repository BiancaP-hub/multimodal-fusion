from tensorflow.keras import layers, models

def multi_scale_features(input_tensor, filters, name_prefix):
    # Path 1: Original scale
    path1 = layers.Conv2D(filters, 3, padding='same', name=name_prefix+'_conv1')(input_tensor)
    path1 = layers.BatchNormalization(name=name_prefix+'_bn1')(path1)
    path1 = layers.ReLU(name=name_prefix+'_relu1')(path1)

    # Path 2: Down-sampled scale
    path2 = layers.Conv2D(filters, 3, strides=2, padding='same', name=name_prefix+'_conv2')(input_tensor)
    path2 = layers.BatchNormalization(name=name_prefix+'_bn2')(path2)
    path2 = layers.ReLU(name=name_prefix+'_relu2')(path2)
    # Upsample to original size
    path2 = layers.UpSampling2D(size=(2, 2), name=name_prefix+'_upsample2')(path2)

    # Combine multi-scale features
    combined_features = layers.Concatenate(axis=-1)([path1, path2])
    return combined_features

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', name=name+'_shortcut')(shortcut)
        shortcut = layers.BatchNormalization(name=name+'_shortcut_bn')(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name+'_conv1')(x)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=name+'_conv2')(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

def build_model(input_shapes, use_multi_scale=True):
    inputs = [layers.Input(shape=shape, name=f'image{i+1}') for i, shape in enumerate(input_shapes)]
    multi_scale_branches = []

    for i, input_tensor in enumerate(inputs):
        if use_multi_scale:
            # Extract multi-scale features
            ms_features = multi_scale_features(input_tensor, 32, name_prefix=f'ms_branch{i+1}')
        else:
            ms_features = input_tensor

        # Pass the combined features through a residual block
        res_block = residual_block(ms_features, 64, name=f'res_block_ms_branch{i+1}')
        multi_scale_branches.append(res_block)

    if len(multi_scale_branches) > 1:
        concatenated_features = layers.Concatenate(axis=-1)(multi_scale_branches)
    else:
        concatenated_features = multi_scale_branches[0]

    # Final layers after fusion
    res_block_concat = residual_block(concatenated_features, 128, name='res_block_concat')
    conv2 = layers.Conv2D(32, 3, strides=1, padding='same', name='conv2')(res_block_concat)
    conv3 = layers.Conv2D(1, 3, strides=1, padding='same', name='conv3')(conv2)

    sigmoid_output = layers.Activation('sigmoid')(conv3)
    weighted_avg_inputs = layers.Average()(inputs + [sigmoid_output])

    model = models.Model(inputs=inputs, outputs=weighted_avg_inputs)
    return model