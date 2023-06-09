import numpy as np

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import BaseGenerator, TARGET_GENERATORS

@TARGET_GENERATORS.register_module
class DEKRGenerator(BaseGenerator):
    
    def __init__(self, num_joints, pos_dist_thresh, bg_weight = 0.1):
        super().__init__()

        self.num_joints = num_joints
        self.pos_dist_thresh = pos_dist_thresh
        self.bg_weight = bg_weight
            
        self.num_joints_with_center = self.num_joints + 1

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def forward(self, annotations, prediction, image_size):

        batch_size, _, output_h, output_w = prediction[0].shape
        output_res = output_h, output_w
        stride_y, stride_x = image_size[0]/output_h, image_size[1]/output_w

        num_joints_without_center = self.num_joints
        num_joints_with_center = num_joints_without_center + 1

        coords = annotations['keypoints'].cpu().numpy()
        area = annotations['area'].cpu().numpy()

        assert self.num_joints + 1 == coords.shape[2], \
            f'the number of joints should be {coords.shape}'
        
        #TODO make it possible to differenciate between center sigma and other sigmas
        scale = max(1/stride_x, 1/stride_y)
        sgm, ct_sgm = (self.pos_dist_thresh/2)*scale, (self.pos_dist_thresh)*scale
        radius = self.pos_dist_thresh*scale

        hms = np.zeros((batch_size, num_joints_with_center, output_h, output_w),
                        dtype=np.float32)
        ignored_hms = 2*np.ones((batch_size, num_joints_with_center, output_h, output_w),
                                    dtype=np.float32)
        offset_map = np.zeros((batch_size, num_joints_without_center*2, output_h, output_w),
                                dtype=np.float32)
        weight_map = np.zeros((batch_size, num_joints_without_center*2, output_h, output_w),
                                dtype=np.float32)
        area_map = np.zeros((batch_size, output_h, output_w),
                            dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for b in range(batch_size):
            for person_id, p in enumerate(coords[b]):
                idx_center = len(p) - 1
                ct_x = int(p[-1, 0])
                ct_y = int(p[-1, 1])
                
                ct_x_sm = (ct_x - stride_x/2)/stride_x
                ct_y_sm = (ct_y - stride_y/2)/stride_y
                for idx, pt in enumerate(p):
                    if idx == idx_center:
                        sigma = ct_sgm
                    else:
                        sigma = sgm
                    if np.any(pt <= 0.):
                        continue
                    x, y = pt[0], pt[1]
                    x_sm, y_sm = (x - stride_x/2)/stride_x, (y - stride_y/2)/stride_y

                    if x_sm < 0 or y_sm < 0 or \
                            x_sm >= output_w or y_sm >= output_h:
                        continue

                    # HEATMAP COMPUTATION
                    ul = int(np.floor(x_sm - 3 * sigma - 1)
                                ), int(np.floor(y_sm - 3 * sigma - 1))
                    br = int(np.ceil(x_sm + 3 * sigma + 2)
                                ), int(np.ceil(y_sm + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], output_res[1])
                    aa, bb = max(0, ul[1]), min(br[1], output_res[0])

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                        cc] = self.get_heat_val(sigma, sx, sy, x_sm, y_sm)

                    hms_list[0][b, idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][b, idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][b, idx, aa:bb, cc:dd] = 1.

                    # OFFSET COMPUTATION
                    if idx != idx_center:
                        start_x = max(int(ct_x_sm - radius), 0)
                        start_y = max(int(ct_y_sm - radius), 0)
                        end_x = min(int(ct_x_sm + radius), output_w)
                        end_y = min(int(ct_y_sm + radius), output_h)

                        for pos_x in range(start_x, end_x):
                            for pos_y in range(start_y, end_y):
                                offset_x = pos_x - x_sm
                                offset_y = pos_y - y_sm
                                if offset_map[b, idx*2, pos_y, pos_x] != 0 \
                                        or offset_map[b, idx*2+1, pos_y, pos_x] != 0:
                                    if area_map[b, pos_y, pos_x] < area[b, person_id]:
                                        continue
                                offset_map[b, idx*2, pos_y, pos_x] = offset_x
                                offset_map[b, idx*2+1, pos_y, pos_x] = offset_y
                                #TODO find a decent constant make weights vary giving animal area
                                weight_map[b, idx*2, pos_y, pos_x] = 1.#/((scale**2)*np.sqrt(area[person_id]))
                                weight_map[b, idx*2+1, pos_y, pos_x] = 1.#/((scale**2)*np.sqrt(area[person_id]))
                                area_map[b, pos_y, pos_x] = area[b, person_id]

        hms_list[1][hms_list[1] == 2] = self.bg_weight

        targets = {
            "heatmaps": hms_list[0],
            "heatmaps_ignored": hms_list[1],
            "locref_maps": offset_map,
            "locref_masks": weight_map
        }
        return targets