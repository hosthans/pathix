class NoTissuePatchesFoundError(RuntimeError):
    pass


class MissingAnnotationPolygonsError(RuntimeError):
    """Raised when annotation-filtered graph construction has no usable polygons."""

    pass


class AnnotationQualitySkipError(RuntimeError):
    """Raised by DynamicalPipeline._check_annotation_quality (regions.py's
    assess_annotation_quality) when a slide's annotations fail the configured QA policy --
    either no annotations at all (CreationConfig.build_slides_without_annotations, default
    False) or annotations present but not spatially matching the slide's own tissue/patches
    (CreationConfig.build_slides_with_mismatched_annotations, default False) -- e.g. the
    GSDB x/y-coordinate-swap bug found 2026-08-12 on 4 scanner-1 MG slides, where real "inflamed"
    polygons existed but matched zero patches in their stored orientation. Distinct from
    MissingAnnotationPolygonsError (that one is about `filter_graph_with_annotations` requiring
    polygons to restrict patch extraction TO; this one is a data-quality gate on whether the
    annotations found are usable AT ALL). Carries structured fields (not just a message string)
    so build_graph.py can write one row per skip to the annotation-QA CSV without re-parsing text.
    """

    def __init__(
        self,
        reason: str,
        slide_name: str,
        n_annotations: int,
        n_patches: int,
        match_rate_normal: float,
        match_rate_swapped: float,
    ) -> None:
        """Build the error with structured QA-skip details for logging."""
        self.reason = reason
        self.slide_name = slide_name
        self.n_annotations = n_annotations
        self.n_patches = n_patches
        self.match_rate_normal = match_rate_normal
        self.match_rate_swapped = match_rate_swapped
        super().__init__(
            f"Annotation QA policy skip for {slide_name}: reason={reason} "
            f"n_annotations={n_annotations} n_patches={n_patches} "
            f"match_rate_normal={match_rate_normal:.4f} match_rate_swapped={match_rate_swapped:.4f}"
        )