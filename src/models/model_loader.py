
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.multimodal import AttentionMIL, MultiModalClassifier
from src.models.base import HistologyClassifier
from src.models.gigapath import GigaPathClassifier

def load_model(model_name: str, feature_extractor: str, num_classes: int=1, dropout_rate: float=0.5):
    """Instantiate a model of the given type with the specified backbone and output settings."""
    match model_name:
        case "attention_mil_multimodal":
            logging.info(f"Loading AttentionMIL with backbone: {feature_extractor}")
            return AttentionMIL(model_name=feature_extractor, num_classes=num_classes, dropout_rate=dropout_rate)
        case "multimodal":
            logging.info(f"Loading MultiModalClassifier with backbone: {feature_extractor}")
            return MultiModalClassifier(model_name=feature_extractor, num_classes=num_classes, dropout_rate=dropout_rate)
        case _:
            logging.info(f"Loading standard HistologyClassifier with backbone: {model_name}")
            return HistologyClassifier(model_name=model_name, num_classes=num_classes, dropout_rate=dropout_rate)



def load_model_with_weights(model_path: Path, device: torch.device, architecture: str = "gigapath") -> nn.Module:
    """Load model based on specified architecture with support for both PyTorch and pickle formats."""

    import timm

    try:
        model = load_model(architecture, feature_extractor="resnet18", num_classes=1)














        file_extension = model_path.suffix.lower()

        if file_extension == '.pkl':
            import pickle
            logging.info(f"Loading pickle model from {model_path}")

            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)


            if hasattr(loaded_data, 'state_dict'):
                model = loaded_data

            else:
                raise ValueError(f"Unsupported pickle format.")

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
                        model.load_state_dict(new_state_dict, strict=False)
                    else:
                        model.load_state_dict(state_dict, strict=False)


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
                        model.load_state_dict(new_state_dict, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
            else:
                raise ValueError(f"Unsupported model format. Expected dictionary with 'model_state_dict'.")


        if not hasattr(model, 'model_name'):
            model.model_name = architecture


        return model.to(device).eval()

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise