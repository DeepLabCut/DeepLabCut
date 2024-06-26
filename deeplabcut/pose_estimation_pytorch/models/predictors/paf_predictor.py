#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    BasePredictor,
    PREDICTORS,
)
from deeplabcut.core import inferenceutils

Graph = list[tuple[int, int]]


@PREDICTORS.register_module
class PartAffinityFieldPredictor(BasePredictor):
    """Predictor class for multiple animal pose estimation with part affinity fields.

    Args:
        num_animals: Number of animals in the project.
        num_multibodyparts: Number of animal's body parts (ignoring unique body parts).
        num_uniquebodyparts: Number of unique body parts.  # FIXME - should not be needed here if we separate the unique bodypart head
        graph: Part affinity field graph edges.
        edges_to_keep: List of indices in `graph` of the edges to keep.
        locref_stdev: Standard deviation for location refinement.
        nms_radius: Radius of the Gaussian kernel.
        sigma: Width of the 2D Gaussian distribution.
        min_affinity: Minimal edge affinity to add a body part to an Assembly.

    Returns:
        Regressed keypoints from heatmaps, locref_maps and part affinity fields, as in Tensorflow maDLC.
    """

    default_init = {
        "locref_stdev": 7.2801,
        "nms_radius": 5,
        "sigma": 1,
        "min_affinity": 0.05,
    }

    def __init__(
        self,
        num_animals: int,
        num_multibodyparts: int,
        num_uniquebodyparts: int,
        graph: Graph,
        edges_to_keep: list[int],
        locref_stdev: float,
        nms_radius: int,
        sigma: float,
        min_affinity: float,
        add_discarded: bool = False,
        apply_sigmoid: bool = True,
        clip_scores: bool = False,
        force_fusion: bool = False,
        return_preds: bool = False,
    ):
        """Initialize the PartAffinityFieldPredictor class.

        Args:
            num_animals: Number of animals in the project.
            num_multibodyparts: Number of animal's body parts (ignoring unique body parts).
            num_uniquebodyparts: Number of unique body parts.
            graph: Part affinity field graph edges.
            edges_to_keep: List of indices in `graph` of the edges to keep.
            locref_stdev: Standard deviation for location refinement.
            nms_radius: Radius of the Gaussian kernel.
            sigma: Width of the 2D Gaussian distribution.
            min_affinity: Minimal edge affinity to add a body part to an Assembly.
            return_preds: Whether to return predictions alongside the animals' poses

        Returns:
            None
        """
        super().__init__()
        self.num_animals = num_animals
        self.num_multibodyparts = num_multibodyparts
        self.num_uniquebodyparts = num_uniquebodyparts
        self.graph = graph
        self.edges_to_keep = edges_to_keep
        self.locref_stdev = locref_stdev
        self.nms_radius = nms_radius
        self.return_preds = return_preds
        self.sigma = sigma
        self.apply_sigmoid = apply_sigmoid
        self.clip_scores = clip_scores
        self.sigmoid = torch.nn.Sigmoid()
        self.assembler = inferenceutils.Assembler.empty(
            num_animals,
            n_multibodyparts=num_multibodyparts,
            n_uniquebodyparts=num_uniquebodyparts,
            graph=graph,
            paf_inds=edges_to_keep,
            min_affinity=min_affinity,
            add_discarded=add_discarded,
            force_fusion=force_fusion,
        )

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of PartAffinityFieldPredictor. Gets predictions from model output.

        Args:
            stride: the stride of the model
            outputs: Output tensors from previous layers.
                output = heatmaps, locref, pafs
                heatmaps: torch.Tensor([batch_size, num_joints, height, width])
                locref: torch.Tensor([batch_size, num_joints, height, width])

        Returns:
            A dictionary containing a "poses" key with the output tensor as value.

        Example:
            >>> predictor = PartAffinityFieldPredictor(num_animals=3, location_refinement=True, locref_stdev=7.2801)
            >>> output = (torch.rand(32, 17, 64, 64), torch.rand(32, 34, 64, 64), torch.rand(32, 136, 64, 64))
            >>> stride = 8
            >>> poses = predictor.forward(stride, output)
        """
        heatmaps = outputs["heatmap"]
        locrefs = outputs["locref"]
        pafs = outputs["paf"]
        scale_factors = stride, stride
        batch_size, n_channels, height, width = heatmaps.shape

        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)

        # Filter predicted heatmaps with a 2D Gaussian kernel as in:
        # https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.pdf
        kernel = self.make_2d_gaussian_kernel(
            sigma=self.sigma, size=self.nms_radius * 2 + 1
        )[None, None]
        kernel = kernel.repeat(n_channels, 1, 1, 1).to(heatmaps.device)
        heatmaps = F.conv2d(
            heatmaps, kernel, stride=1, padding="same", groups=n_channels
        )

        peaks = self.find_local_peak_indices_maxpool_nms(
            heatmaps, self.nms_radius, threshold=0.01
        )
        if ~torch.any(peaks):
            return {"poses": torch.zeros((batch_size, 0, self.num_multibodyparts, 5))}

        locrefs = locrefs.reshape(batch_size, n_channels, 2, height, width)
        locrefs = locrefs * self.locref_stdev
        pafs = pafs.reshape(batch_size, -1, 2, height, width)

        graph = [self.graph[ind] for ind in self.edges_to_keep]
        preds = self.compute_peaks_and_costs(
            heatmaps,
            locrefs,
            pafs,
            peaks,
            graph,
            self.edges_to_keep,
            scale_factors,
            n_id_channels=0,  # FIXME Handle identity training
        )
        poses = torch.empty((batch_size, self.num_animals, self.num_multibodyparts, 5))
        poses_unique = torch.empty((batch_size, 1, self.num_uniquebodyparts, 4))
        for i, data_dict in enumerate(preds):
            assemblies, unique = self.assembler._assemble(data_dict, ind_frame=0)
            for j, assembly in enumerate(assemblies):
                poses[i, j, :, :4] = torch.from_numpy(assembly.data)
                poses[i, j, :, 4] = assembly.affinity
            if unique is not None:
                poses_unique[i, 0, :, :4] = torch.from_numpy(unique)

        if self.clip_scores:
            poses[..., 2] = torch.clip(poses[..., 2], min=0, max=1)

        out = {"poses": poses}
        if self.return_preds:
            out["preds"] = preds
        return out

    @staticmethod
    def find_local_peak_indices_maxpool_nms(
        input_: torch.Tensor, radius: int, threshold: float
    ) -> torch.Tensor:
        pooled = F.max_pool2d(input_, kernel_size=radius, stride=1, padding=radius // 2)
        maxima = input_ * torch.eq(input_, pooled).float()
        peak_indices = torch.nonzero(maxima >= threshold, as_tuple=False)
        return peak_indices.int()

    @staticmethod
    def make_2d_gaussian_kernel(sigma: float, size: int) -> torch.Tensor:
        k = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32) ** 2
        k = F.softmax(-k / (2 * (sigma ** 2)), dim=0)
        return torch.einsum("i,j->ij", k, k)

    @staticmethod
    def calc_peak_locations(
        locrefs: torch.Tensor,
        peak_inds_in_batch: torch.Tensor,
        strides: tuple[float, float],
        n_decimals: int = 3,
    ) -> torch.Tensor:
        s, b, r, c = peak_inds_in_batch.T
        stride_y, stride_x = strides
        strides = torch.Tensor((stride_x, stride_y)).to(locrefs.device)
        off = locrefs[s, b, :, r, c]
        loc = strides * peak_inds_in_batch[:, [3, 2]] + strides // 2 + off
        return torch.round(loc, decimals=n_decimals)

    @staticmethod
    def compute_edge_costs(
        pafs: NDArray,
        peak_inds_in_batch: NDArray,
        graph: Graph,
        paf_inds: list[int],
        n_bodyparts: int,
        n_points: int = 10,
        n_decimals: int = 3,
    ) -> list[dict[int, NDArray]]:
        # Clip peak locations to PAFs dimensions
        h, w = pafs.shape[-2:]
        peak_inds_in_batch[:, 2] = np.clip(peak_inds_in_batch[:, 2], 0, h - 1)
        peak_inds_in_batch[:, 3] = np.clip(peak_inds_in_batch[:, 3], 0, w - 1)

        n_samples = pafs.shape[0]
        sample_inds = []
        edge_inds = []
        all_edges = []
        all_peaks = []
        for i in range(n_samples):
            samples_i = peak_inds_in_batch[:, 0] == i
            peak_inds = peak_inds_in_batch[samples_i, 1:]
            if not np.any(peak_inds):
                continue
            peaks = peak_inds[:, 1:]
            bpt_inds = peak_inds[:, 0]
            idx = np.arange(peaks.shape[0])
            idx_per_bpt = {j: idx[bpt_inds == j].tolist() for j in range(n_bodyparts)}
            edges = []
            for k, (s, t) in zip(paf_inds, graph):
                inds_s = idx_per_bpt[s]
                inds_t = idx_per_bpt[t]
                if not (inds_s and inds_t):
                    continue
                candidate_edges = ((i, j) for i in inds_s for j in inds_t)
                edges.extend(candidate_edges)
                edge_inds.extend([k] * len(inds_s) * len(inds_t))
            if not edges:
                continue
            sample_inds.extend([i] * len(edges))
            all_edges.extend(edges)
            all_peaks.append(peaks[np.asarray(edges)])
        if not all_peaks:
            return [dict() for _ in range(n_samples)]

        sample_inds = np.asarray(sample_inds, dtype=np.int32)
        edge_inds = np.asarray(edge_inds, dtype=np.int32)
        all_edges = np.asarray(all_edges, dtype=np.int32)
        all_peaks = np.concatenate(all_peaks)
        vecs_s = all_peaks[:, 0]
        vecs_t = all_peaks[:, 1]
        vecs = vecs_t - vecs_s
        lengths = np.linalg.norm(vecs, axis=1).astype(np.float32)
        lengths += np.spacing(1, dtype=np.float32)
        xy = np.linspace(vecs_s, vecs_t, n_points, axis=1, dtype=np.int32)
        y = pafs[
            sample_inds.reshape((-1, 1)),
            edge_inds.reshape((-1, 1)),
            :,
            xy[..., 0],
            xy[..., 1],
        ]
        integ = np.trapz(y, xy[..., ::-1], axis=1)
        affinities = np.linalg.norm(integ, axis=1).astype(np.float32)
        affinities /= lengths
        np.round(affinities, decimals=n_decimals, out=affinities)
        np.round(lengths, decimals=n_decimals, out=lengths)

        # Form cost matrices
        all_costs = []
        for i in range(n_samples):
            samples_i_mask = sample_inds == i
            costs = dict()
            for k in paf_inds:
                edges_k_mask = edge_inds == k
                idx = np.flatnonzero(samples_i_mask & edges_k_mask)
                s, t = all_edges[idx].T
                n_sources = np.unique(s).size
                n_targets = np.unique(t).size
                costs[k] = dict()
                costs[k]["m1"] = affinities[idx].reshape((n_sources, n_targets))
                costs[k]["distance"] = lengths[idx].reshape((n_sources, n_targets))
            all_costs.append(costs)

        return all_costs

    @staticmethod
    def _linspace(start: torch.Tensor, stop: torch.Tensor, num: int) -> torch.Tensor:
        # Taken from https://github.com/pytorch/pytorch/issues/61292#issue-937937159
        steps = torch.linspace(0, 1, num, dtype=torch.float32, device=start.device)
        steps = steps.reshape([-1, *([1] * start.ndim)])
        out = start[None] + steps * (stop - start)[None]
        return out.swapaxes(0, 1)

    def compute_peaks_and_costs(
        self,
        heatmaps: torch.Tensor,
        locrefs: torch.Tensor,
        pafs: torch.Tensor,
        peak_inds_in_batch: torch.Tensor,
        graph: Graph,
        paf_inds: list[int],
        strides: tuple[float, float],
        n_id_channels: int,
        n_points: int = 10,
        n_decimals: int = 3,
    ) -> list[dict[str, NDArray]]:
        n_samples, n_channels = heatmaps.shape[:2]
        n_bodyparts = n_channels - n_id_channels
        pos = self.calc_peak_locations(locrefs, peak_inds_in_batch, strides, n_decimals)
        pos = pos.detach().cpu().numpy()
        heatmaps = heatmaps.detach().cpu().numpy()
        pafs = pafs.detach().cpu().numpy()
        peak_inds_in_batch = peak_inds_in_batch.detach().cpu().numpy()
        costs = self.compute_edge_costs(
            pafs, peak_inds_in_batch, graph, paf_inds, n_bodyparts, n_points, n_decimals
        )
        s, b, r, c = peak_inds_in_batch.T
        prob = np.round(heatmaps[s, b, r, c], n_decimals).reshape((-1, 1))
        if n_id_channels:
            ids = np.round(heatmaps[s, -n_id_channels:, r, c], n_decimals)

        peaks_and_costs = []
        for i in range(n_samples):
            xy = []
            p = []
            id_ = []
            samples_i_mask = peak_inds_in_batch[:, 0] == i
            for j in range(n_bodyparts):
                bpts_j_mask = peak_inds_in_batch[:, 1] == j
                idx = np.flatnonzero(samples_i_mask & bpts_j_mask)
                xy.append(pos[idx])
                p.append(prob[idx])
                if n_id_channels:
                    id_.append(ids[idx])
            dict_ = {"coordinates": (xy,), "confidence": p}
            if costs is not None:
                dict_["costs"] = costs[i]
            if n_id_channels:
                dict_["identity"] = id_
            peaks_and_costs.append(dict_)

        return peaks_and_costs
