import argparse
import re
from datetime import datetime
from pathlib import Path


def prefix_arg(value: str) -> str | None:
    """argparse `type=` for --prefix / --eval_prefix."""
    try:
        return PathConfigurator.normalize_prefix(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


class PathConfigurator:

    @staticmethod
    def normalize_prefix(prefix) -> str | None:
        """Filesystem-safe run prefix, or None when unset ('' / 'none').

        Letters, digits, '.', '_' and '-' are kept; any other run of characters becomes '-'.
        """
        if prefix is None:
            return None
        text = str(prefix).strip()
        if not text or text.lower() == "none":
            return None
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.")
        if not safe:
            raise ValueError(f"Invalid prefix {prefix!r}: use letters, digits, '.', '_' or '-'.")
        return safe

    @staticmethod
    def _get_field(obj, name):
        return obj.get(name, None) if isinstance(obj, dict) else getattr(obj, name, None)

    @staticmethod
    def get_prefix(obj) -> str | None:
        """Prefix of the run itself (args or meta dict): meta, checkpoints, history, logs, training dir."""
        return PathConfigurator.normalize_prefix(PathConfigurator._get_field(obj, "prefix"))

    @staticmethod
    def get_eval_prefix(obj) -> str | None:
        """Prefix for evaluation outputs: --eval_prefix if given, else the run's own prefix."""
        eval_prefix = PathConfigurator.normalize_prefix(PathConfigurator._get_field(obj, "eval_prefix"))
        return eval_prefix or PathConfigurator.get_prefix(obj)

    @staticmethod
    def apply_prefix(name: str, prefix) -> str:
        """`<prefix>_<name>`; unchanged when there is no prefix."""
        prefix = PathConfigurator.normalize_prefix(prefix)
        return f"{prefix}_{name}" if prefix else name

    @staticmethod
    def _config_token(config_name) -> str | None:
        if not config_name or str(config_name).strip().lower() == "none":
            return None
        return Path(str(config_name)).stem

    @staticmethod
    def _resolve_dataset(meta, config=None) -> str:
        if isinstance(meta, dict):
            dataset = meta.get("dataset", None)
        else:
            dataset = getattr(meta, "dataset", None)
        if not dataset and isinstance(config, dict):
            dataset = config.get("dataset")
        return dataset or "gsdb"

    @staticmethod
    def _resolve_config_name(meta, config=None) -> str | None:
        if isinstance(meta, dict):
            config_name = meta.get("config_name", None)
        else:
            config_name = getattr(meta, "config_name", None)
        if not config_name and isinstance(config, dict):
            config_name = config.get("config_name")
        return config_name

    @staticmethod
    def get_meta_name(meta, config=None, with_timestamp: bool = False, timestamp: str = None) -> str:
        """Build a consistent meta filename from the run's relevant attributes."""
        # Descriptive attributes form the (truncatable) middle of the name; flag attributes and the
        # timestamp form the tail, which -- like the task/heonly head -- is never truncated.
        descriptive_attrs = [
            "model_type", "architecture", "input_type", "embedding_type", "pipeline", "feature_extractor",
        ]
        flag_attrs = [
            "mil", "cached", "training_method", "fine_tune", "split_phase", "test_split",
            "cv_fold",
        ]

        def _attr_tokens(attrs):
            tokens = []
            for attr in attrs:
                val = meta.get(attr, None) if isinstance(meta, dict) else getattr(meta, attr, None)
                if isinstance(val, bool):
                    if val:
                        tokens.append(f"{attr}")
                elif val is not None and str(val).lower() != "none":
                    tokens.append(f"{attr}-{val}")
            return tokens

        # A run prefix (--prefix) leads the name; it sits in `head`, so it is never truncated below.
        run_prefix = PathConfigurator.get_prefix(meta)
        parts = [run_prefix] if run_prefix else []

        # Task leads the filename (matches _build_name's checkpoint/log naming below) so it's
        # visible immediately in a directory listing instead of buried behind the cfg- token.
        # cfg- follows right after task-, ahead of dataset-, since task alone doesn't distinguish
        # runs (e.g. every GSDB run is task-inflammation regardless of whether it's tissue_type or
        # segmentation; every camelyon run is task-lymph) -- cfg- is what actually identifies the
        # specific setting, so it needs to be visible without reading past dataset- first.
        task = meta.get("task", None) if isinstance(meta, dict) else getattr(meta, "task", None)
        if task is not None and str(task).lower() != "none":
            parts.append(f"task-{task}")

        # Data-restriction flags go right after task- (not in attr_names at the end): long cfg-
        # tokens push the name past max_len below, and the truncation would silently cut them off.
        for attr, token in (("he_only", "heonly"), ("scanner1_only", "scanner1only")):
            val = meta.get(attr, False) if isinstance(meta, dict) else getattr(meta, attr, False)
            if val:
                parts.append(token)

        head = parts
        middle = []
        config_token = PathConfigurator._config_token(PathConfigurator._resolve_config_name(meta, config))
        if config_token:
            middle.append(f"cfg-{config_token}")
        middle.append(f"dataset-{PathConfigurator._resolve_dataset(meta, config)}")
        middle.extend(_attr_tokens(descriptive_attrs))

        tail = _attr_tokens(flag_attrs)
        if with_timestamp:
            if not timestamp:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tail.append(str(timestamp))

        name = "_".join(head + middle + tail) or "meta"

        # Most filesystems (ext4 included) cap a single filename component at 255 bytes.
        # MetaProcessor.save_meta appends ".yaml" to this, and a descriptive config_name (long
        # cfg- token) can push the assembled name past that -- which crashes save_meta *after*
        # training finishes, right as it's about to hand off to the automatic post-training
        # evaluation in train_model.py's _run_training (real cause: `OSError: [Errno 36] File name
        # too long`, though a bare `except Exception` in save_meta previously re-raised that as a
        # misleading "MemoryError" -- fixed alongside this). Only the descriptive middle is
        # shortened (plus a short hash of the full name so two long names can't collide) -- head
        # (task, heonly/scanner1only) and tail (flags, timestamp) always stay readable.
        max_len = 200  # leaves margin for MetaProcessor's ".yaml" suffix and typical 255-byte caps
        if len(name.encode("utf-8")) > max_len:
            import hashlib
            digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            fixed = "_".join(head + [digest] + tail)
            budget = max_len - len(fixed.encode("utf-8")) - 1  # "_" joining the shortened middle
            middle_str = "_".join(middle)[:max(budget, 0)].rstrip("_")
            name = "_".join(head + ([middle_str] if middle_str else []) + [digest] + tail)
        return name

    @staticmethod
    def _build_name(args, config=None, include_test_split: bool = False, with_timestamp: bool = False,
                    use_eval_prefix: bool = False) -> str:
        dataset = PathConfigurator._resolve_dataset(args, config)
        config_token = PathConfigurator._config_token(PathConfigurator._resolve_config_name(args, config))

        parts = [
            getattr(args, "task", None),
            f"dataset-{dataset}",
            f"cfg-{config_token}" if config_token else None,
            getattr(args, "model_type", None),
            getattr(args, "architecture", None),
            getattr(args, "input_type", None),
            getattr(args, "embedding_type", None),
            getattr(args, "pipeline", None),
            getattr(args, "feature_extractor", None),
        ]
        if getattr(args, "he_only", False):
            parts.append("heonly")
        if getattr(args, "scanner1_only", False):
            parts.append("scanner1only")

        split_phase = getattr(args, "split_phase", None)
        if split_phase and str(split_phase).strip().lower() != "none":
            parts.append(str(split_phase).strip().lower())

        fine_tuning_enabled = bool(getattr(args, "fine_tune", False))
        if not fine_tuning_enabled:
            model_config = getattr(args, "model_config", None)
            if isinstance(model_config, dict):
                fine_tuning_cfg = model_config.get("fine_tuning", {})
                if isinstance(fine_tuning_cfg, dict):
                    fine_tuning_enabled = bool(fine_tuning_cfg.get("enabled", False))
        if not fine_tuning_enabled and isinstance(config, dict):
            fine_tuning_cfg = config.get("fine_tuning", {})
            if isinstance(fine_tuning_cfg, dict):
                fine_tuning_enabled = bool(fine_tuning_cfg.get("enabled", False))
        if fine_tuning_enabled:
            parts.append("finetune")

        cv_fold = getattr(args, "cv_fold", None)
        if cv_fold is not None and str(cv_fold).strip().lower() != "none":
            parts.append(f"cvfold-{cv_fold}")

        # Only tag the name with seed when it's a deliberate override away from the config's own
        # declared seed (e.g. training a seed-ensemble with --seed 43/44/... against a config that
        # bakes in seed: 42) -- every normal run (args.seed == config's seed) keeps the exact same
        # checkpoint/log filename as before this was added, so nothing existing breaks. Without
        # this, two runs that only differ by --seed silently overwrite the same checkpoint file.
        run_seed = getattr(args, "seed", None)
        config_seed = config.get("seed") if isinstance(config, dict) else None
        if run_seed is not None and config_seed is not None and int(run_seed) != int(config_seed):
            parts.append(f"seed-{run_seed}")

        name = "_".join([str(p) for p in parts if p and p != "none"])
        if include_test_split:
            test_split = getattr(args, "test_split", None)
            if test_split:
                name = f"{test_split}_{name}"
        if with_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{timestamp}"
        prefix = PathConfigurator.get_eval_prefix(args) if use_eval_prefix else PathConfigurator.get_prefix(args)
        return PathConfigurator.apply_prefix(name or "run", prefix)

    @staticmethod
    def get_logging_path(args, paths, config=None):
        """Return the log directory path for a run."""
        run_name = PathConfigurator._build_name(args, config=config, with_timestamp=True)
        return paths["LOGS_DIR"] / run_name

    @staticmethod
    def get_checkpoint_path(args, paths, config=None):
        """Return the best-checkpoint file path for a run."""
        run_name = PathConfigurator._build_name(args, config=config, with_timestamp=False)
        return paths["MODELS_DIR"] / f"best_model_{run_name}.pt"

    @staticmethod
    def get_periodic_checkpoint_path(args, paths, config, epoch, tag: str = None):
        """Return the periodic checkpoint file path for a run and epoch."""
        run_name = PathConfigurator._build_name(args, config=config, with_timestamp=False)
        tag_part = f"_{tag}" if tag else ""
        return paths["MODELS_DIR"] / f"checkpoint_{run_name}{tag_part}_epoch{epoch}.pt"

    @staticmethod
    def get_training_history_path(args, paths, config, tag: str = None):
        """Return the training history file path for a run."""
        run_name = PathConfigurator._build_name(args, config=config, with_timestamp=False)
        tag_part = f"_{tag}" if tag else ""
        return paths["MODELS_DIR"] / f"training_history_{run_name}{tag_part}.json"

    @staticmethod
    def get_evaluation_path(args, paths, config=None):
        """Return the evaluation output directory path for a run."""
        run_name = PathConfigurator._build_name(
            args, config=config, include_test_split=True, with_timestamp=True, use_eval_prefix=True,
        )
        return paths["RESULTS_DIR"] / "evaluations" / run_name

    @staticmethod
    def get_training_path(args, paths, config=None):
        """Return the training output directory path for a run."""
        run_name = PathConfigurator._build_name(args, config=config, with_timestamp=True)
        return paths["MODELS_DIR"] / run_name
