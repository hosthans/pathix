"""
Common model utility functions to break circular dependencies.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import logging

def get_transforms(is_training: bool = False) -> transforms.Compose:
    """Get data transformations matching GigaPath's requirements."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))
        ])
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    ])

def _log_state_dict_load_result(result: "torch.nn.modules.module._IncompatibleKeys", model_path: Path) -> None:
    if result.missing_keys:
        logging.warning(
            "Checkpoint %s is missing %d expected parameter(s) (left at random init): %s",
            model_path,
            len(result.missing_keys),
            result.missing_keys,
        )
    if result.unexpected_keys:
        logging.warning(
            "Checkpoint %s has %d unexpected parameter(s) not present in the model (ignored): %s",
            model_path,
            len(result.unexpected_keys),
            result.unexpected_keys,
        )
    if not result.missing_keys and not result.unexpected_keys:
        logging.info("Checkpoint %s matched the model exactly (no missing/unexpected keys).", model_path)


def infer_checkpoint_in_dim(
    model_path: Path,
    candidate_keys: tuple = ("input_proj.weight", "patch_in.weight"),
) -> "int | None":
    """Recover the input feature dim a checkpoint was trained with by reading its saved input-projection weight shape."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        return None
    if not isinstance(state_dict, dict):
        return None

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {(k[7:] if str(k).startswith("module.") else k): v for k, v in state_dict.items()}

    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[1])
    return None


def load_model(
    model_path: Path,
    model: nn.Module,
    device: torch.device,
    architecture: str = "gigapath",
    strict: bool = False,
) -> nn.Module:
    """Load model based on specified architecture with support for both PyTorch and pickle formats."""

    try:

        file_extension = model_path.suffix.lower()

        if file_extension == '.pkl':
            import pickle
            logging.info(f"Loading pickle model from {model_path}")

            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)


            if hasattr(loaded_data, 'state_dict'):
                model = loaded_data
            elif isinstance(loaded_data, dict):

                if 'model_state_dict' in loaded_data:
                    result = model.load_state_dict(loaded_data['model_state_dict'], strict=strict)
                    _log_state_dict_load_result(result, model_path)
                else:

                    result = model.load_state_dict(loaded_data, strict=strict)
                    _log_state_dict_load_result(result, model_path)
            else:
                raise ValueError(f"Unsupported pickle format. Expected model or state dict.")

        else:
            logging.info(f"Loading PyTorch model from {model_path}")
            try:

                checkpoint = torch.load(model_path, map_location=device)
            except Exception as e:
                logging.warning(f"Failed to load with default settings, trying with weights_only=False: {str(e)}")

                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                logging.info("Successfully loaded model with weights_only=False")


            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:

                    state_dict = checkpoint['model_state_dict']


                    if any(k.startswith('module.') for k in state_dict.keys()):
                        logging.info("Detected DataParallel saved model, removing 'module.' prefix")
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:] if k.startswith('module.') else k
                            new_state_dict[name] = v
                        result = model.load_state_dict(new_state_dict, strict=strict)
                        _log_state_dict_load_result(result, model_path)
                    else:
                        result = model.load_state_dict(state_dict, strict=strict)
                        _log_state_dict_load_result(result, model_path)


                    if 'trial_number' in checkpoint:
                        logging.info(f"Loaded model from trial {checkpoint['trial_number']}")
                    if 'val_loss' in checkpoint:
                        logging.info(f"Validation loss: {checkpoint['val_loss']:.4f}")


                    if 'epoch' in checkpoint and checkpoint['epoch'] != -1:
                        logging.info(f"Trained for {checkpoint['epoch']} epochs")
                else:

                    logging.warning("No 'model_state_dict' key found, attempting to load as direct state dict")


                    if any(k.startswith('module.') for k in checkpoint.keys()):
                        logging.info("Detected DataParallel saved model, removing 'module.' prefix")
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in checkpoint.items():
                            name = k[7:] if k.startswith('module.') else k
                            new_state_dict[name] = v
                        result = model.load_state_dict(new_state_dict, strict=strict)
                        _log_state_dict_load_result(result, model_path)
                    else:
                        result = model.load_state_dict(checkpoint, strict=strict)
                        _log_state_dict_load_result(result, model_path)
            else:
                raise ValueError(f"Unsupported model format. Expected dictionary with 'model_state_dict'.")


        if not hasattr(model, 'model_name'):
            model.model_name = architecture


        return model.to(device).eval()

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise