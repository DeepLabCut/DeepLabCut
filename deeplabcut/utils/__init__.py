#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Internal helpers for DeepLabCut.

``submod_attrs`` below preserves the flat namespace that the previous
``from .<module> import *`` lines produced -- now resolved lazily.

TODO @deruyter92 2026-08-31: We should consider removing these imports altogether
and stop advertising a public API for them. But this would be a breaking change.
see https://github.com/DeepLabCut/DeepLabCut/pull/3459
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "auxfun_models",
        "auxfun_multianimal",
        "auxfun_videos",
        "auxiliaryfunctions",
        "auxiliaryfunctions_3d",
        "conversioncode",
        "frameselectiontools",
        "make_labeled_video",
        "multiprocessing",
        "pandas_future_mode",
        "plotting",
        "pseudo_label",
        "skeleton",
        "video_processor",
        "visualization",
    ],
    # TODO @deruyter92 2026-09-03: This is a hack to preserve the flat
    # namespace that the previous ``from .<module> import *`` lines produced.
    # We should consider removing (some of) these as public API.
    submod_attrs={
        "auxfun_multianimal": [
            "IntersectionofIndividualsandOnesGivenbyUser",
            "LoadFullMultiAnimalData",
            "SaveFullMultiAnimalData",
            "check_inferencecfg_sanity",
            "convert2_maDLC",
            "convert_single2multiplelegacyAM",
            "extractindividualsandbodyparts",
            "filter_unwanted_paf_connections",
            "form_default_inferencecfg",
            "get_track_method",
            "getpafgraph",
            "graph2names",
            "prune_paf_graph",
            "read_inferencecfg",
            "reorder_individuals_in_df",
            "returnlabelingdata",
            "validate_paf_graph",
        ],
        "auxfun_videos": [
            "CropVideo",
            "DEFAULT_EXCLUDE_PATTERNS",
            "DownSampleVideo",
            "SUPPORTED_VIDEOS",
            "ShortenVideo",
            "VideoReader",
            "VideoWriter",
            "check_video_integrity",
            "collect_video_paths",
            "draw_bbox",
            "imread",
            "imresize",
            "rotate_video",
        ],
        "auxiliaryfunctions": [
            "CheckifNotAnalyzed",
            "CheckifNotEvaluated",
            "CheckifPostProcessing",
            "GetDataandMetaDataFilenames",
            "GetEvaluationFolder",
            "GetModelFolder",
            "GetScorerName",
            "GetTrainingSetFolder",
            "GetVideoList",
            "IntersectionofBodyPartsandOnesGivenbyUser",
            "LoadMetadata",
            "SaveData",
            "SaveMetadata",
            "attempt_to_make_folder",
            "check_if_not_analyzed",
            "check_if_not_evaluated",
            "check_if_post_processing",
            "create_config_template",
            "create_config_template_3d",
            "edit_config",
            "filter_files_by_patterns",
            "find_analyzed_data",
            "find_next_unlabeled_folder",
            "find_video_full_data",
            "find_video_metadata",
            "form_data_containers",
            "get_bodyparts",
            "get_data_and_metadata_filenames",
            "get_deeplabcut_path",
            "get_evaluation_folder",
            "get_labeled_data_folder",
            "get_list_of_videos",
            "get_model_folder",
            "get_scorer_name",
            "get_snapshot_index_for_scorer",
            "get_snapshots_from_folder",
            "get_training_set_folder",
            "get_unique_bodyparts",
            "get_video_list",
            "grab_files_in_folder",
            "intersection_of_body_parts_and_ones_given_by_user",
            "load_analyzed_data",
            "load_detection_data",
            "load_metadata",
            "load_video_full_data",
            "load_video_metadata",
            "read_config",
            "read_pickle",
            "read_plainconfig",
            "safe_resolve",
            "save_data",
            "save_metadata",
            "write_config",
            "write_config_3d",
            "write_config_3d_template",
            "write_pickle",
            "write_plainconfig",
        ],
        "conversioncode": [
            "SUPPORTED_FILETYPES",
            "adapt_labeled_data_to_new_project",
            "analyze_videos_converth5_to_csv",
            "analyze_videos_converth5_to_nwb",
            "convertcsv2h5",
            "guarantee_multiindex_rows",
            "merge_windowsannotationdataONlinuxsystem",
        ],
        "frameselectiontools": [
            "KmeansbasedFrameselection",
            "KmeansbasedFrameselectioncv2",
            "UniformFrames",
            "UniformFramescv2",
        ],
        "make_labeled_video": [
            "CreateVideo",
            "CreateVideoSlow",
            "create_labeled_video",
            "create_video",
            "create_video_from_pickled_tracks",
            "create_video_with_all_detections",
            "create_video_with_keypoints_only",
            "get_segment_indices",
            "proc_video",
        ],
        "plotting": [
            "Histogram",
            "PlottingResults",
            "plot_edge_affinity_distributions",
            "plot_trajectories",
        ],
        "video_processor": [
            "VideoProcessor",
            "VideoProcessorCV",
        ],
    },
)
