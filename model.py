import segmentation_models_pytorch as smp


def get_model(model_name="UnetPlusPlus", encoder_name="resnet34"):
    model_class = getattr(smp, model_name)

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    return model
