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
from collections import defaultdict

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
        heatmaps = outputs["heatmap"]  # (batch_size, num_joints, height, width)
        locrefs = outputs["locref"]  # (batch_size, num_joints*2, height, width)
        pafs = outputs["paf"]  # (batch_size, num_edges*2, height, width)
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
        )  # (n_peaks, 4) -> columns: (batch, part, height, width)
        if ~torch.any(peaks):
            poses = -torch.ones(
                (batch_size, self.num_animals, self.num_multibodyparts, 5)
            )
            results = dict(poses=poses)
            if self.return_preds:
                results["preds"] = ([dict(coordinates=[[]], costs=[])],)

            return results

        locrefs = locrefs.reshape(batch_size, n_channels, 2, height, width)
        locrefs = (
            locrefs * self.locref_stdev
        )  # (batch_size, num_joints, 2, height, width)
        pafs = pafs.reshape(
            batch_size, -1, 2, height, width
        )  # (batch_size, num_edges, 2, height, width)

        # Use only the minimal tree edges for efficiency
        graph = [self.graph[ind] for ind in self.edges_to_keep]
        # Compute refined peak coords + PAF line-integral costs
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
        # Initialize output tensors
        poses = -torch.ones(
            (batch_size, self.num_animals, self.num_multibodyparts, 5)
        )  # (batch_size, num_animals, num_joints, [x, y, prob, id, affinity])
        poses_unique = -torch.ones(
            (batch_size, 1, self.num_uniquebodyparts, 4)
        )  # (batch_size, 1, num_unique_joints, [x, y, prob, id])
        # Greedy bipartite assembly per frame
        for i, data_dict in enumerate(preds):
            assemblies, unique = self.assembler._assemble(data_dict, ind_frame=0)
            if assemblies is not None:
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
        k = F.softmax(-k / (2 * (sigma**2)), dim=0)
        return torch.einsum("i,j->ij", k, k)

    @staticmethod
    def calc_peak_locations(
        locrefs: torch.Tensor,
        peak_inds_in_batch: torch.Tensor,
        strides: tuple[float, float],
    ) -> torch.Tensor:
        """Refine peak coordinates to input-image pixels using locrefs and stride."""
        s, b, r, c = peak_inds_in_batch.T
        stride_y, stride_x = strides
        strides = torch.Tensor((stride_x, stride_y)).to(locrefs.device)
        off = locrefs[s, b, :, r, c]
        loc = strides * peak_inds_in_batch[:, [3, 2]] + strides // 2 + off
        return loc

    @staticmethod
    def compute_edge_costs(
        pafs: torch.Tensor,
        peak_inds: torch.Tensor,
        graph: Graph,
        paf_limb_inds: list[int],
        n_bodyparts: int,
        n_points: int = 10,
        n_decimals: int = 3,
    ) -> list[dict[int, NDArray]]:
        """Compute PAF line-integral affinities per limb.

        Args:
            pafs: Part Affinity Fields tensor with shape (batch_size, num_edges, 2, height, width).
                Contains vector fields representing limb orientations between body parts.
            peak_inds: Peak indices array with shape (n_peaks, 4) containing
                [batch_index, bodypart_index, row, col] for each detected peak.
            graph: List of tuples representing edges in the pose graph. Each tuple contains
                (source_bodypart_index, target_bodypart_index).
            paf_limb_inds: List of indices specifying which edges from the graph to use for
                PAF computation. Length should match the number of PAF channels.
            n_bodyparts: Total number of body parts in the pose model.
            n_points: Number of points to sample along each limb segment for PAF integration.
                Default is 10.
            n_decimals: Number of decimal places to round affinity and distance values.
                Default is 3.

        Returns:
            List of per-image cost dictionaries, one for each image in the batch. Each
            dictionary maps PAF edge indices to cost matrices with keys:
                - "m1": Affinity matrix with shape (n_source_peaks, n_target_peaks)
                - "distance": Distance matrix with shape (n_source_peaks, n_target_peaks)
        """
        device = pafs.device
        graph = torch.tensor(graph, dtype=torch.long, device=device).T  # (2, num_edges)
        paf_limb_inds = torch.tensor(paf_limb_inds, dtype=torch.long, device=device)
        batch_size = pafs.shape[0]
        h, w = pafs.shape[-2:]

        # Clip peak locations to PAF map bounds
        peak_inds[:, 2] = torch.clamp(peak_inds[:, 2], 0, h - 1)
        peak_inds[:, 3] = torch.clamp(peak_inds[:, 3], 0, w - 1)

        peak_batches = peak_inds[:, 0]
        peak_bodyparts = peak_inds[:, 1]
        peak_rows = peak_inds[:, 2]
        peak_cols = peak_inds[:, 3]

        src_bodypart_id = graph[0]  # (n_edges,)
        dst_bodypart_id = graph[1]  # (n_edges,)

        # Process each batch separately to reduce memory usage
        all_edge_idx = []
        all_src_idx = []
        all_dst_idx = []
        all_batch_inds = []

        for batch_idx in range(batch_size):
            # Get peaks for this batch only
            batch_mask = peak_batches == batch_idx
            if not torch.any(batch_mask):
                continue

            batch_peak_indices = torch.nonzero(batch_mask, as_tuple=False).squeeze(-1)
            batch_bodyparts = peak_bodyparts[batch_mask]

            # Masks of peaks that match each edge's source/dest bodypart for this batch
            src_mask = batch_bodyparts.unsqueeze(0) == src_bodypart_id.unsqueeze(
                1
            )  # (n_edges, n_batch_peaks)
            dst_mask = batch_bodyparts.unsqueeze(0) == dst_bodypart_id.unsqueeze(
                1
            )  # (n_edges, n_batch_peaks)

            # Valid src/dst peaks for each edge in this batch: (n_edges, n_batch_peaks, n_batch_peaks)
            valid_pairs = src_mask.unsqueeze(2) & dst_mask.unsqueeze(1)

            # Indices of all valid pairs for this batch
            edge_idx, src_idx, dst_idx = valid_pairs.nonzero(as_tuple=True)

            if len(edge_idx) > 0:
                # Map back to original peak indices
                src_idx = batch_peak_indices[src_idx]
                dst_idx = batch_peak_indices[dst_idx]

                all_edge_idx.append(edge_idx)
                all_src_idx.append(src_idx)
                all_dst_idx.append(dst_idx)
                all_batch_inds.append(torch.full_like(edge_idx, batch_idx))

        if not all_edge_idx:
            return [{} for _ in range(batch_size)]

        # Concatenate results from all batches
        edge_idx = torch.cat(all_edge_idx)
        src_idx = torch.cat(all_src_idx)
        dst_idx = torch.cat(all_dst_idx)
        batch_inds = torch.cat(all_batch_inds)

        edge_idx = paf_limb_inds[edge_idx]  # Map back to original PAF indices

        # Gather coordinates
        src_coords = torch.stack(
            [peak_rows[src_idx], peak_cols[src_idx]], dim=1
        )  # (found_pairs, 2)
        dst_coords = torch.stack(
            [peak_rows[dst_idx], peak_cols[dst_idx]], dim=1
        )  # (found_pairs, 2)

        vecs_s = src_coords.float()  # (found_pairs, 2)
        vecs_t = dst_coords.float()  # (found_pairs, 2)
        vecs = vecs_t - vecs_s
        lengths = torch.norm(vecs, dim=1)
        lengths += torch.tensor(np.spacing(1, dtype=np.float32), device=device)

        # Sample n_points along the segments
        t_vals = torch.linspace(0, 1, n_points, device=device, dtype=torch.float32)
        t_vals = t_vals.unsqueeze(0).unsqueeze(-1)  # (1, n_points, 1)

        # Interpolate points along each segment: (n_edges, n_points, 2)
        xy = vecs_s.unsqueeze(1) + t_vals * (vecs_t - vecs_s).unsqueeze(1)
        xy = xy.to(torch.int32)
        xy[..., 0] = torch.clamp(xy[..., 0], 0, h - 1)
        xy[..., 1] = torch.clamp(xy[..., 1], 0, w - 1)

        # Gather PAF vectors at sampled pixels: (n_edges, n_points, 2)
        y = pafs[
            batch_inds.unsqueeze(1).expand(-1, n_points),  # (n_edges, n_points)
            edge_idx.unsqueeze(1).expand(-1, n_points),  # (n_edges, n_points)
            :,  # both x and y components of each vector
            xy[..., 0],  # row coordinates
            xy[..., 1],  # col coordinates
        ]

        # Integrate PAF along segment using trapezoidal rule
        xy_reversed = torch.flip(
            xy.float(), dims=[-1]
        )
        integ = torch.trapz(y, xy_reversed, dim=1)  # (n_edges, 2)
        affinities = torch.norm(integ, dim=1)  # (n_edges,)
        affinities = affinities / lengths
        affinities = torch.round(affinities * (10**n_decimals)) / (10**n_decimals)
        lengths = torch.round(lengths * (10**n_decimals)) / (10**n_decimals)

        edge_idx = edge_idx.cpu().numpy()
        src_idx = src_idx.cpu().numpy()
        dst_idx = dst_idx.cpu().numpy()
        batch_inds = batch_inds.cpu().numpy()
        affinities = affinities.cpu().numpy()
        lengths = lengths.cpu().numpy()
        paf_limb_inds = paf_limb_inds.cpu().numpy()

        # Form per-image, per-limb cost matrices for bipartite matching
        order = np.lexsort((edge_idx, batch_inds))
        batch_inds = batch_inds[order]
        edge_idx = edge_idx[order]
        src_idx = src_idx[order]
        dst_idx = dst_idx[order]
        affinities = affinities[order]
        lengths = lengths[order]

        # Run-length encode on (batch, limb) boundaries where (batch, limb) changes
        change = np.empty(batch_inds.size, dtype=bool)
        change[0] = True
        change[1:] = (batch_inds[1:] != batch_inds[:-1]) | (
            edge_idx[1:] != edge_idx[:-1]
        )
        group_starts = np.flatnonzero(change)
        # Add sentinel end
        group_ends = np.r_[group_starts[1:], batch_inds.size]

        # Build an index dict of slices for group lookup by (batch, limb)
        batch_groups = defaultdict(list)  # (batch)->list of (limb, start, end)
        for st, en in zip(group_starts, group_ends):
            b = batch_inds[st].item()
            k = edge_idx[st].item()
            batch_groups[b].append((k, st, en))

        paf_limb_inds = set(paf_limb_inds.tolist())

        all_costs = []
        for b in range(batch_size):
            costs = {}
            # find this batch's groups
            for k, st, en in batch_groups.get(b, []):
                if k not in paf_limb_inds:
                    continue
                s = src_idx[st:en]
                t = dst_idx[st:en]

                n_s = np.unique(s).size
                n_t = np.unique(t).size

                m1 = affinities[st:en].reshape((n_s, n_t))
                dist = lengths[st:en].reshape((n_s, n_t))

                costs[k] = {"m1": m1, "distance": dist}

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
        paf_limb_inds: list[int],
        strides: tuple[float, float],
        n_id_channels: int,
        n_points: int = 10,
        n_decimals: int = 3,
    ) -> list[dict[str, NDArray]]:
        """
        Compute refined peak coordinates, confidence scores, and PAF edge costs for pose estimation.

        Args:
            heatmaps: Smoothed heatmaps tensor with shape (batch_size, num_joints, height, width).
                Contains confidence scores for each body part at each spatial location.
            locrefs: Location refinement maps with shape (batch_size, num_joints, 2, height, width).
                Contains sub-pixel offset corrections for precise keypoint localization.
            pafs: Part Affinity Fields tensor with shape (batch_size, num_edges, 2, height, width).
                Contains vector fields representing limb orientations between body parts.
            peak_inds_in_batch: Peak indices tensor with shape (n_peaks, 4) containing
                [batch_index, bodypart_index, row, col] for each detected peak.
            graph: List of tuples representing edges in the pose graph. Each tuple contains
                (source_bodypart_index, target_bodypart_index).
            paf_limb_inds: List of indices specifying which edges from the graph to use for
                PAF computation. Length should match the number of PAF channels.
            strides: Tuple of (stride_y, stride_x) representing the downsampling factor
                from input image to feature maps.
            n_id_channels: Number of identity channels in the heatmaps for individual
                identification. These channels are located at the end of the heatmap tensor.
            n_points: Number of points to sample along each limb segment for PAF integration.
                Default is 10.
            n_decimals: Number of decimal places to round coordinate and confidence values.
                Default is 3.

        Returns:
            List of dictionaries, one per image in the batch. Each dictionary contains:
                - "coordinates": Tuple containing a list of numpy arrays, one per body part.
                    Each array has shape (n_peaks_for_bodypart, 2) with [x, y] coordinates.
                - "confidence": List of numpy arrays, one per body part. Each array has shape
                    (n_peaks_for_bodypart, 1) containing confidence scores.
                - "costs": (Optional) Cost matrix for PAF edge connections between body parts.
                    Only present if PAF computation is successful.
                - "identity": (Optional) List of numpy arrays containing identity features,
                    one per body part. Only present if n_id_channels > 0.
        """
        batch_size, n_channels = heatmaps.shape[:2]
        n_bodyparts = n_channels - n_id_channels
        # Refine peak positions to input-image pixels
        pos = self.calc_peak_locations(
            locrefs, peak_inds_in_batch, strides
        )  # (n_peaks, 2)

        # Compute per-limb affinity matrices via PAF line integral
        costs = self.compute_edge_costs(
            pafs,
            peak_inds_in_batch,
            graph,
            paf_limb_inds,
            n_bodyparts,
            n_points,
            n_decimals,
        )
        s, b, r, c = peak_inds_in_batch.T
        # Extract confidence at each peak from smoothed heatmap
        prob = heatmaps[s, b, r, c].unsqueeze(-1)
        if n_id_channels:
            ids = heatmaps[s, -n_id_channels:, r, c]

        peak_inds_in_batch = peak_inds_in_batch.cpu().numpy()
        peaks_and_costs = []
        pos = pos.cpu().numpy()
        prob = prob.cpu().numpy()
        for batch_idx in range(batch_size):
            xy = []
            p = []
            id_ = []
            samples_i_mask = peak_inds_in_batch[:, 0] == batch_idx
            for j in range(n_bodyparts):
                bpts_j_mask = peak_inds_in_batch[:, 1] == j
                idx = np.flatnonzero(samples_i_mask & bpts_j_mask)
                xy.append(pos[idx])
                p.append(prob[idx])
                if n_id_channels:
                    id_.append(ids[idx])
            dict_ = {"coordinates": (xy,), "confidence": p}
            if costs is not None:
                dict_["costs"] = costs[batch_idx]
            if n_id_channels:
                dict_["identity"] = id_
            peaks_and_costs.append(dict_)

        return peaks_and_costs

    def set_paf_edges_to_keep(self, edge_indices: list[int]) -> None:
        """Sets the PAF edge indices to use to assemble individuals

        Args:
            edge_indices: The indices of edges in the graph to keep.
        """
        self.edges_to_keep = edge_indices
        self.assembler.paf_inds = edge_indices
