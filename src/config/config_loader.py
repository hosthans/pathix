from pathlib import Path
import logging
import yaml
import json
import sys


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def load_config_new(
    config_dir: Path,
    task: str = "inflammation",
    mil: bool = False,
    architecture: str = "resnet18",
    dataset: str = "gsdb",
    config_name: str = None,
) -> dict:
    """Load task-specific and model-specific configuration from unified YAML file."""
    if config_name:
        candidate = Path(config_name)
        if candidate.is_file():
            candidate_paths = [candidate]
        else:
            if candidate.suffix:
                candidate_paths = [config_dir / candidate]
            else:
                candidate_paths = [config_dir / f"{candidate.name}.yaml"]
    else:
        candidate_paths = [
            config_dir / 'model_config_mil.yaml',
            config_dir / 'model_config.yaml',
        ] if mil else [
            config_dir / 'model_config.yaml',
            config_dir / 'model_config_mil.yaml',
        ]

    last_error = None
    for config_path in candidate_paths:
        logging.info(f"Loading config from {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)


            common_config = config.get('common', {})

            resolved_task = task
            logging.info("Resolving task name for config lookup... configured task: %s", task)
            if resolved_task not in config:
                raise ValueError(f"Task '{task}' not found in config file. Please check the task name or update the config file.")

            if resolved_task not in config:
                raise KeyError(f"Task '{task}' not found in config file")

            task_config = config[resolved_task]

            if architecture is None:
                raise ValueError(f"No model architecture specified. Please provide a model architecture for task '{task}'")

            if architecture not in task_config:
                raise KeyError(
                    f"Model architecture '{architecture}' not found for task '{task}' in {config_path.name}. "
                    f"Available architectures: {list(task_config.keys())}"
                )

            model_config = task_config[architecture]

            dataset_settings = config.get('dataset_settings', {})
            dataset_config = dataset_settings.get(dataset, {})
            if task in dataset_config and isinstance(dataset_config.get(task), dict):
                dataset_config = dataset_config.get(task, {})
            elif resolved_task in dataset_config and isinstance(dataset_config.get(resolved_task), dict):
                dataset_config = dataset_config.get(resolved_task, {})

            result_config = common_config.copy()
            result_config.update(dataset_config)
            result_config.update(model_config)


            result_config['architectures'] = config.get('architectures', {})


            result_config['task'] = task
            result_config['selected_model'] = architecture
            result_config['dataset'] = dataset
            result_config['config_name'] = str(config_path)


            logging.info(f"Loaded config: {json.dumps(result_config, indent=2, default=str)}")
            return result_config

        except FileNotFoundError:
            last_error = FileNotFoundError(f"Config file not found: {config_path}")
            continue
        except (KeyError, ValueError) as e:
            last_error = e
            continue

    if last_error is not None:
        logging.error(str(last_error))
        raise last_error
    raise FileNotFoundError("No valid config file candidates could be resolved")


def load_config(config_dir: Path, task: str, model_name: str = None) -> dict:
    """Load task-specific and model-specific configuration from unified YAML file."""
    config_path = config_dir / 'model_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)


        common_config = config.get('common', {})


        if task not in config:
            raise KeyError(f"Task '{task}' not found in config file")

        task_config = config[task]


        if model_name is None:
            model_name = next(iter(task_config.keys()))
            logging.info(f"No model specified. Using default model: {model_name}")


        if model_name not in task_config:
            available_models = list(task_config.keys())
            raise KeyError(f"Model '{model_name}' not found for task '{task}'. Available models: {available_models}")

        model_config = task_config[model_name]


        result_config = common_config.copy()
        result_config.update(model_config)


        result_config['architectures'] = config.get('architectures', {})


        result_config['task'] = task
        result_config['selected_model'] = model_name


        logging.info(f"Loaded config: {json.dumps(result_config, indent=2, default=str)}")

        return result_config

    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
