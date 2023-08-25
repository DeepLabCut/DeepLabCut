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

"""Definition for official DeepLabCut benchmark tasks.

See benchmark.deeplabcut.org for a current leaderboard with models and metrics
for each of these benchmarks. Submissions can be done by opening a PR in the
benchmark reporistory:

https://github.com/DeepLabCut/benchmark
"""

import deeplabcut.benchmark.base


class TriMouseBenchmark(deeplabcut.benchmark.base.Benchmark):
    """Datasets with three mice with a top-view camera.

    Three wild-type (C57BL/6J) male mice ran on a paper spool following odor trails (Mathis et al 2018). These experiments were carried out in the laboratory of Venkatesh N. Murthy at Harvard University. Data were recorded at 30 Hz with 640 x 480 pixels resolution acquired with a Point Grey Firefly FMVU-03MTM-CS. One human annotator was instructed to localize the 12 keypoints (snout, left ear, right ear, shoulder, four spine points, tail base and three tail points). All surgical and experimental procedures for mice were in accordance with the National Institutes of Health Guide for the Care and Use of Laboratory Animals and approved by the Harvard Institutional Animal Care and Use Committee. 161 frames were labeled, making this a real-world sized laboratory dataset.

    Introduced in Lauer et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19, no. 4 (2022): 496-504.
    """

    name = "trimouse"
    keypoints = (
        "snout",
        "leftear",
        "rightear",
        "shoulder",
        "spine1",
        "spine2",
        "spine3",
        "spine4",
        "tailbase",
        "tail1",
        "tail2",
        "tailend",
    )
    ground_truth = deeplabcut.benchmark.get_filepath("CollectedData_Daniel.h5")
    metadata = deeplabcut.benchmark.get_filepath(
        "Documentation_data-MultiMouse_70shuffle1.pickle"
    )
    num_animals = 3


class ParentingMouseBenchmark(deeplabcut.benchmark.base.Benchmark):
    """Datasets with three mice, one parenting, two pups.

    Parenting behavior is a pup directed behavior observed in adult mice involving complex motor actions directed towards the benefit of the offspring. These experiments were carried out in the laboratory of Catherine Dulac at Harvard University. The behavioral assay was performed in the homecage of singly housed adult female mice in dark/red light conditions. For these videos, the adult mice was monitored for several minutes in the cage followed by the introduction of pup (4 days old) in one corner of the cage. The behavior of the adult and pup was monitored for a duration of 15 minutes. Video was recorded at 30Hz using a Microsoft LifeCam camera (Part#: 6CH-00001) with a resolution of 1280 x 720 pixels or a Geovision camera (model no.: GV-BX4700-3V) also acquired at 30 frames per second at a resolution of 704 x 480 pixels. A human annotator labeled on the adult animal the same 12 body points as in the tri-mouse dataset, and five body points on the pup along its spine. Initially only the two ends were labeled, and intermediate points were added by interpolation and their positions was manually adjusted if necessary. All surgical and experimental procedures for mice were in accordance with the National Institutes of Health Guide for the Care and Use of Laboratory Animals and approved by the Harvard Institutional Animal Care and Use Committee. 542 frames were labeled, making this a real-world sized laboratory dataset.

    Introduced in Lauer et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19, no. 4 (2022): 496-504.
    """

    name = "parenting"
    keypoints = (
        "end1",
        "interm1",
        "interm2",
        "interm3",
        "end2",
        "snout",
        "leftear",
        "rightear",
        "shoulder",
        "spine1",
        "spine2",
        "spine3",
        "spine4",
        "tailbase",
        "tail1",
        "tail2",
        "tailend",
    )

    ground_truth = deeplabcut.benchmark.get_filepath("CollectedData_Mostafizur.h5")
    metadata = deeplabcut.benchmark.get_filepath(
        "Documentation_data-CrackingParenting_70shuffle1.pickle"
    )
    num_animals = 2

    def compute_pose_map(self, results_objects):
        return deeplabcut.benchmark.metrics.calc_map_from_obj(
            results_objects,
            h5_file=self.ground_truth,
            metadata_file=self.metadata,
            oks_sigma=0.15,
            margin=10,
            symmetric_kpts=[(0, 4), (1, 3)],
        )


class MarmosetBenchmark(deeplabcut.benchmark.base.Benchmark):
    """Dataset with two marmosets.

    All animal procedures are overseen by veterinary staff of the MIT and Broad Institute Department of Comparative Medicine, in compliance with the NIH guide for the care and use of laboratory animals and approved by the MIT and Broad Institute animal care and use committees. Video of common marmosets (Callithrix jacchus) was collected in the laboratory of Guoping Feng at MIT. Marmosets were recorded using Kinect V2 cameras (Microsoft) with a resolution of 1080p and frame rate of 30 Hz. After acquisition, images to be used for training the network were manually cropped to 1000 x 1000 pixels or smaller. The dataset is 7,600 labeled frames from 40 different marmosets collected from 3 different colonies (in different facilities). Each cage contains a pair of marmosets, where one marmoset had light blue dye applied to its tufts. One human annotator labeled the 15 marker points on each animal present in the frame (frames contained either 1 or 2 animals).

    Introduced in Lauer et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19, no. 4 (2022): 496-504.
    """

    name = "marmosets"
    keypoints = (
        "Front",
        "Right",
        "Middle",
        "Left",
        "FL1",
        "BL1",
        "FR1",
        "BR1",
        "BL2",
        "BR2",
        "FL2",
        "FR2",
        "Body1",
        "Body2",
        "Body3",
    )
    ground_truth = deeplabcut.benchmark.get_filepath("CollectedData_Mackenzie.h5")
    metadata = deeplabcut.benchmark.get_filepath(
        "Documentation_data-Marmoset_70shuffle1.pickle"
    )
    num_animals = 2


class FishBenchmark(deeplabcut.benchmark.base.Benchmark):
    """Dataset with multiple fish, filmed from top-view

    Schools of inland silversides (Menidia beryllina, n=14 individuals per school) were recorded in the Lauder Lab at Harvard University while swimming at 15 speeds (0.5 to 8 BL/s, body length, at 0.5 BL/s intervals) in a flow tank with a total working section of 28 x 28 x 40 cm as described in previous work, at a constant temperature (18±1°C) and salinity (33 ppt), at a Reynolds number of approximately 10,000 (based on BL). Dorsal views of steady swimming across these speeds were recorded by high-speed video cameras (FASTCAM Mini AX50, Photron USA, San Diego, CA, USA) at 60-125 frames per second (feeding videos at 60 fps, swimming alone 125 fps). The dorsal view was recorded above the swim tunnel and a floating Plexiglas panel at the water surface prevented surface ripples from interfering with dorsal view videos. Five keypoints were labeled (tip, gill, peduncle, dorsal fin tip, caudal tip). 100 frames were labeled, making this a real-world sized laboratory dataset.

    Introduced in Lauer et al. "Multi-animal pose estimation, identification and tracking with DeepLabCut." Nature Methods 19, no. 4 (2022): 496-504.
    """

    name = "fish"
    keypoints = ("tip", "gill", "peduncle", "caudaltip", "dfintip")
    ground_truth = deeplabcut.benchmark.get_filepath("CollectedData_Valentina.h5")
    metadata = deeplabcut.benchmark.get_filepath(
        "Documentation_data-Schooling_70shuffle1.pickle"
    )
    num_animals = 14

    def compute_pose_rmse(self, results_objects):
        return deeplabcut.benchmark.metrics.calc_rmse_from_obj(
            results_objects,
            h5_file=self.ground_truth,
            metadata_file=self.metadata,
            drop_kpts=[4, 5],
        )

    def compute_pose_map(self, results_objects):
        return deeplabcut.benchmark.metrics.calc_map_from_obj(
            results_objects,
            h5_file=self.ground_truth,
            metadata_file=self.metadata,
            drop_kpts=[4, 5],
        )
