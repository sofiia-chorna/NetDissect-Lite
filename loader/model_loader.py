from collections import OrderedDict
import settings
import torch
import torchvision


def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        # Load a pretrained model from torchvision
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        # Load checkpoint
        checkpoint = torch.load(settings.MODEL_FILE)

        if isinstance(checkpoint, (OrderedDict, dict)):
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)

            # Remove 'module.' prefix if needed (in parallel mode)
            if settings.MODEL_PARALLEL:
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

    # Register hooks for feature extraction
    for name in settings.FEATURE_NAMES:
        if name in model._modules:
            model._modules.get(name).register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Feature layer {name} not found in the model!")

    if settings.GPU:
        model = model.cuda()

    model.eval()
    return model
