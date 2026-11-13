from pathlib import Path
from typing import Dict, List, Optional

import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.enums import AnnotationType, MaskType

class PathResolver:
    def __init__(self, paths: Dict[str, Path]):
        """Store the project path mapping used to resolve concrete file paths."""
        self._paths = paths

    def _get(self, key: str) -> Path:
        if key not in self._paths:
            raise RuntimeError(f"Missing path key: {key}")
        return self._paths[key]

    @staticmethod
    def slide_id(slide_path: Path) -> str:
        """Return the slide identifier derived from its file stem."""
        return slide_path.stem

    def slide_path(
        self,
        dataset: str,
        target_directory: str,
        slide_name: str,
        extensions: Optional[List[str]] = None
    ) -> Path:
        """Resolve the on-disk path of a slide by name, searching known extensions."""

        base = self._get("RAW_DIR") / dataset / target_directory

        if extensions is None:
            extensions = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]

        for ext in extensions:
            candidate = base / f"{slide_name}{ext}"
            if candidate.exists():
                return candidate

        matches = []
        for ext in extensions:
            matches.extend(base.rglob(f"{slide_name}{ext}"))

        if not matches:
            raise FileNotFoundError(
                f"No slide found for '{slide_name}' in {base}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple slides found for '{slide_name}': {matches}"
            )

        return matches[0]


    def annotation_path(
        self,
        slide_path: Path,
        dataset: str,
        annotation_type: AnnotationType,
        annotation_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Resolve the annotation file path for a slide given its annotation type."""
        base = Path(annotation_dir) if annotation_dir is not None else self._get("RAW_DIR") / dataset / "annotations"
        slide_id = self.slide_id(slide_path)

        if annotation_type == AnnotationType.ASAP:
            return base / f"{slide_id}.xml"
        elif annotation_type == AnnotationType.GEOJSON:
            return base / f"{slide_id}_annotations.json"
        else:
            print(f"Unsupported annotation type: {annotation_type}")
            return None


    def thumbnail_path(self, slide_path: Path, dataset: str, size: int) -> Path:
        """Resolve the thumbnail file path for a slide."""
        base = self._get("PROCESSED_DIR") / dataset / "thumbnails"
        return base / f"{self.slide_id(slide_path)}_{size}.png"

    def mask_path(
        self,
        slide_path: Path,
        dataset: str,
        directory: str = None,
        mask_type: MaskType = MaskType.SLIDE,
        mask_dir: Optional[Path] = None,
    ) -> Path:
        """Resolve the mask file path for a slide."""
        if mask_dir is not None:
            base = Path(mask_dir)
        elif directory is None:
            base = self._get("RAW_DIR") / dataset / "masks"
        else:
            base = self._get("RAW_DIR") / dataset / directory
        return base / f"{self.slide_id(slide_path)}_mask.png" if mask_type == MaskType.IMAGE else base / f"{self.slide_id(slide_path)}_mask.tif"

    def embedding_path(self, slide_path: Path, dataset: str, model_name: str) -> Path:
        """Resolve the embedding file path for a slide."""
        base = self._get("EMBEDDING_DIR") / dataset / model_name
        return base / f"{self.slide_id(slide_path)}.pt"

    def graph_path(self, slide_path: Path, dataset: str, pipeline_name: str) -> Path:
        """Resolve the graph file path for a slide."""
        base = self._get("GRAPH_DIR") / dataset / pipeline_name
        return base / f"{self.slide_id(slide_path)}.pt"