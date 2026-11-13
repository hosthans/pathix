from enum import Enum


class SlideType(str, Enum):
    MRXS = "mrxs"
    TIFF = "tiff"

class AnnotationType(str, Enum):
    ASAP = "asap"
    GEOJSON = "geojson"

class MaskType(str, Enum):
    SLIDE = "slide"
    IMAGE = "image"

class DatasetType(str, Enum):
    CAMELYON17 = "camelyon17"
    GSDB = "gsdb"


class NodeCreationMode(str, Enum):
    PATCH_ENCODER = "patch_encoder"
    CELL_SEGMENTATION = "cell_segmentation"


class EdgeConstructionMode(str, Enum):
    KNN = "knn"
    DISTANCE = "distance"
    TARGETHEAD_INTERSECTION = "targethead_intersection"
    SEMANTIC_EQUALITY = "semantic_equality"