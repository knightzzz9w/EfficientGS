import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer_mip import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians, eval_ate_all
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.logging_utils import TicToc
from utils.descriptor import GlobalDesc


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.netvlad = GlobalDesc()

        self.submap_lc_info = {}
        self.first_viewpoint = None
        self.first_viewpoint_occ = None
        self.last_kf_viewpoint = None
        self.last_kf_visbility = None
        self.rot_thre = config["Training"]["rot_thre"]
        self.trans_thre = config["Training"]["trans_thre"]
        self.submap_desc = []
        self.keyframes_info = []
        self.submap_id = 0
        self.lc_sim = config["Training"]["lc_sim"]
        self.if_last_kf = False
        self.max_exceed_rot = config["Training"]["max_exceed_rot"]
        self.submap_rots_diff = {}
        self.min_interval = config["Training"]["min_interval"]

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.opacity_threshold = config["Training"]["opacity_threshold"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.kernel_size = config["Training"]["kernel_size"]
        #self.skipstep = config["Training"]["skipstep"]

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False,  opacity_threshold=0.7):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        front_depth_threshold = self.config["Training"]["front_depth_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if not init:
            valid_rgb2 = torch.logical_and(valid_rgb, opacity < opacity_threshold)
            depth_mask  = torch.from_numpy((viewpoint.depth > front_depth_threshold)[None])
            valid_rgb2 = torch.logical_and(valid_rgb2, depth_mask.cuda())
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        if not init:
            initial_depth_origin = torch.from_numpy(viewpoint.depth).unsqueeze(0)  #for viewpoint change
            initial_depth = torch.from_numpy(viewpoint.depth.copy()).unsqueeze(0)

            initial_depth_origin[~valid_rgb.cpu()] = 0  # for viewpoint change
            initial_depth[~valid_rgb2.cpu()] = 0

        else:
            initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
            initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()  #new submap

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.first_viewpoint = viewpoint
        self.keyframes_info.append(0)
        self.submap_rots_diff[0] = []
        self.submap_rots_diff[0].append(torch.tensor(0.0).cuda())

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - 1]
        # # print("prev R and T is " , prev.R , prev.T)
        T_k_1 = torch.eye(4).to(self.device)
        T_k_1[:3, :3] = prev.R_gt ; T_k_1[:3, 3] = prev.T_gt
        T_k = torch.eye(4).to(self.device)
        T_k[:3, :3] = viewpoint.R_gt ; T_k[:3, 3] = viewpoint.T_gt

        T_k_1k = torch.inverse(T_k_1)@T_k
        T_Wk_1 = torch.eye(4).to(self.device)
        T_Wk_1[:3, :3] = self.cameras[cur_frame_idx - 1].R ; T_Wk_1[:3, 3] = self.cameras[cur_frame_idx - 1].T 
        T_Wk = T_Wk_1@T_k_1k
        viewpoint.update_RT(T_Wk[:3, :3], T_Wk[:3, 3])
        t1 = time.time()

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        #print("begin tracking and cur gaussian size is " , self.gaussians._xyz.shape)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, self.kernel_size
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
            if tracking_itr % 30 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket_mip(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                #print("Tracking converged and actual tracking iters is " , tracking_itr )
                break


        self.median_depth = get_median_depth(depth, opacity)  #mean depth sort by opacity
        
        time.sleep(max(0.4 - (time.time() - t1) , 0.0))
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]  #check every gaussian
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame 

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
 
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility


        if data[0] == "keyframe":
            self.last_kf_visbility = list(self.occ_aware_visibility.items())[0]

        current_win = [kf_id for kf_id , _ , _  in keyframes]

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())


        if  self.first_viewpoint is not None:
            if len(current_win) == 0:
                #render
                with torch.no_grad():
                    render_pkg = render(
                        self.first_viewpoint, self.gaussians, self.pipeline_params, self.background, self.kernel_size
                    )
                    n_touched  =  render_pkg["n_touched"]
                    self.first_viewpoint_occ = (0 , n_touched)  
            
            
            elif self.first_viewpoint.uid not in current_win :
                with torch.no_grad():
                    render_pkg = render(
                        self.first_viewpoint, self.gaussians, self.pipeline_params, self.background, self.kernel_size
                    )
                    n_touched  =  render_pkg["n_touched"]
                    self.first_viewpoint_occ = (self.first_viewpoint.uid , n_touched)

            else:
                self.first_viewpoint_occ =  (self.first_viewpoint.uid , occ_aware_visibility[self.first_viewpoint.uid])
        


    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def update_submaps_info(self):
        with torch.no_grad():
            submap_desc = torch.cat(self.submap_desc)
            self_sim = torch.einsum("id,jd->ij", submap_desc, submap_desc)
            score_min, _ = self_sim.topk(max(int(len(submap_desc) * self.lc_sim), 1))  #submap each frame with len*sim sim frame in this submap
            
        self.submap_lc_info[self.submap_id] = {
                "submap_id": self.submap_id,
                "kf_id": np.array(self.keyframes_info),
                "kf_desc": submap_desc,
                "self_sim": score_min, # per image self similarity within the submap
            }
        
        self.submap_desc = []
        
        
    def refine_render_submap(self):
        #turn list to torch tensor
        submap_rots_diff = torch.stack(self.submap_rots_diff[self.submap_id])
        #find rots > 20 degree than first keyframe and 
        exceed_rot = submap_rots_diff > self.max_exceed_rot
        exceed_rot = exceed_rot
        exceed_kf_ids = torch.tensor(self.keyframes_info).cuda()[exceed_rot]
        #print("exceed_kf_ids is " , exceed_kf_ids)

        #render all exceed rots keyframes in 100 iters , every iter randomoly choose 8 frames, and update the gaussians ;  all process in backend
        msg = ["submap", exceed_kf_ids]
        self.backend_queue.put(msg)

        
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        while True:
            while self.frontend_queue.empty():
                time.sleep(0.05)
            data = self.frontend_queue.get()
            if data[0] == "submap":
                break
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        self.occ_aware_visibility = occ_aware_visibility

    def detect_closure(self, query_id: int, final=False):
        
        n_submaps = self.submap_id + 1
        query_info = self.submap_lc_info[query_id]
        iterator = range(query_id+1, n_submaps) if final else range(query_id)
        db_info_list = [self.submap_lc_info[i] for i in iterator]  #query to n_submaps
        db_desc_map_id = []
        for db_info in db_info_list:
            db_desc_map_id += [db_info['submap_id'] for _ in db_info['kf_desc']]  #all keyframe ids
        db_desc_map_id = torch.Tensor(db_desc_map_id).to(self.device)
        
        query_desc = query_info['kf_desc']
        db_desc = torch.cat([db_info['kf_desc'] for db_info in db_info_list])
        
        with torch.no_grad():
            cross_sim = torch.einsum("id,jd->ij", query_desc, db_desc)  #query.len * db.len
            self_sim = query_info['self_sim']  #query.len * len2
            matches = torch.argwhere(cross_sim > self_sim[:,[-1]])[:,-1]  # > submap best ids
            matched_map_ids = db_desc_map_id[matches].long().unique()   #one submap id for loop closure
        
        # filter out invalid matches
        filtered_mask = abs(matched_map_ids - query_id) > self.min_interval
        matched_map_ids = matched_map_ids[filtered_mask]  #query matched to cur map id
        print("matched_map_ids is " , matched_map_ids)

        return matched_map_ids


    def judge_submap(self, viewpoint , cur_visibility):
        
        last_idx = self.first_viewpoint.uid
        delta_pose = torch.linalg.inv(viewpoint.T_cw)@self.first_viewpoint.T_cw  
        translation_diff = torch.norm(delta_pose[:3, 3])
        rot_euler_diff_deg = torch.arccos((torch.trace(delta_pose[:3, :3]) - 1)*0.5) * 180 / 3.1415926
        rot_euler_diff_deg = torch.abs(rot_euler_diff_deg)
        print("cur frame id is " , viewpoint.uid  ,  "kf id is " , cur_visibility[0] , " and first  viewpoint id is " , self.first_viewpoint_occ[0] )
        intersection = torch.logical_and(
            cur_visibility[1], self.first_viewpoint_occ[1]
        ).count_nonzero()



        # union = torch.logical_or(
        #     cur_visibility, self.first_viewpoint_occ
        # ).count_nonzero()
        
        union = self.first_viewpoint_occ[1].shape[0]

        ratio = intersection / union
        print("translation_diff is " , translation_diff , "rot_euler_diff_deg is " , rot_euler_diff_deg ,  "and ratio is " , ratio)
            
        if self.submap_id not in self.submap_rots_diff.keys():
            self.submap_rots_diff[self.submap_id] = []

        self.submap_rots_diff[self.submap_id].append(rot_euler_diff_deg)
        
        #exceeds_thresholds = (translation_diff > self.trans_thre) or rot_euler_diff_deg > self.rot_thre or ratio < 0.4

        exceeds_thresholds = (translation_diff > self.trans_thre) or rot_euler_diff_deg > self.rot_thre  # and we need to sync render viewpoints
        if exceeds_thresholds:
            self.first_viewpoint = None
            #self.refine_render_submap()
            self.update_submaps_info()
            if self.submap_id > 0:
                self.detect_closure(self.submap_id)
            self.submap_id += 1
            self.keyframes_info = []
            print("start new submap")

    
    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tracking_time = TicToc()
                tracking_time.tic()
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                # if self.single_thread and self.requested_keyframe > 0:
                #     time.sleep(0.01)
                #     continue

                if  self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                #self.last_kf_viewpoint = 

                if self.if_last_kf:
                    self.if_last_kf = False
                    self.judge_submap(self.last_kf_viewpoint , self.last_kf_visbility)
                    self.last_kf_viewpoint = None


                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix 
                )
                viewpoint.compute_grad_mask(self.config)

                #print("Frontend !!! Prepare viewpoint and 3D point cost " , tracking_time.toc()  ,"ms")

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.last_kf_viewpoint = viewpoint #last viewpoint
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                #viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)


                #print("Frontend !!! Tracking and refining exposure cost " , tracking_time.toc()  ,"ms")

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket_mip(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                #print("Frontend !!! Push to UI cost " , tracking_time.toc()  ,"ms")

                if self.requested_keyframe > 0:   #request backend opt
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf

                # print("Frontend !!! Judge Kf cost " , tracking_time.toc()  ,"ms")
                # print("current window is " , [cur_win for cur_win in self.current_window])

                if create_kf:

                    self.if_last_kf = True
                    self.last_kf_viewpoint = viewpoint   #for new judge submap update

                    
                    if self.first_viewpoint is None:
                        self.first_viewpoint = viewpoint
                    #self.last_kf_visbility = curr_visibility

                    self.keyframes_info.append(cur_frame_idx)
                    cur_submap_desc = self.netvlad(viewpoint.original_image.unsqueeze(0))
                    self.submap_desc.append(cur_submap_desc)
                    #self.judge_submap(viewpoint , curr_visibility)
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                        opacity_threshold=self.opacity_threshold,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    # print("Frontend !!! Create Kf cost " , tracking_time.toc()  ,"ms")
                    # print("current gaussian size  is " , self.gaussians._xyz.shape)
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                    eval_ate_all(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                # if create_kf:
                #     # throttle at 3fps when keyframe is added
                #     duration = tic.elapsed_time(toc)
                #     time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
