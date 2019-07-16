from models.unets import resnet152_fpn, resnet101_fpn, resnet50_fpn, xception_fpn,  densenet_fpn, inception_resnet_v2_fpn


def make_model(network, input_shape,output_channels,chosen_activation):
    if network == 'resnet101_softmax':
        return resnet101_fpn(input_shape,channels=output_channels, activation=chosen_activation)
    elif network == 'resnet152_2':
        return resnet152_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnet101_2':
        return resnet101_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnetv2':
        return inception_resnet_v2_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnetv2_3':
        return inception_resnet_v2_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'densenet169':
        return densenet_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'densenet169_softmax':
        return densenet_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnet101_unet_2':
        return resnet101_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'xception_fpn':
        return xception_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=output_channels, activation=chosen_activation)
    else:
        raise ValueError('unknown network ' + network)
