import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping , get_loss_tracking , get_loss_mapping_rgbd2
from utils.logging_utils import TicToc
from utils.convert_stamp import stamp2seconds , stamp2seconds_header
import numpy as np
from scipy.spatial.transform import Rotation

def odom2pose(odom):

    position = odom.pose.position  #R_wbk
    orientation = odom.pose.orientation   #T_wbk
    
    translation = np.array([position.x, position.y, position.z])
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    
    T_k = np.eye(4)
    T_k[ :3 ,  :3 ] = rotation_matrix ; T_k[ :3 , 3 ] = translation

    T_k = torch.from_numpy((T_k).astype(float)).to("cuda")

    return T_k

def view_queue(q):
    elements = []
    size = q.qsize()  # 获取队列的大小
    for _ in range(size):
        element = q.get()
        elements.append(element)
        q.put(element)  # 取出后再放回队列
    return elements

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        #self.recent_viewpoints = {}
        self.current_window = []
        self.T_bc = None
        #self.recent_windows = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        #self.recentframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    
    def add_next_kf2(self, frame_idx, viewpoint,  ext_points, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq2(
            viewpoint, ext_points , kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,  #3 std gaussian in the screen
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            if mapping_iteration == 2:
                print("init step and render depth mean is " , depth.mean())
                print("init step and render depth min is " , depth.min())
                print("init step and render depth max is " , depth.max())
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.init_densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1 , map_recent = False):
        if len(current_window) == 0:
            return
        t1 = time.time() 
        map_time = TicToc() ; map_time.tic()
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        # print("current window is " , current_window)
        # print("recent window is " , self.recent_windows)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        if map_recent:
            iters_recent = self.config["Training"]["newframe_init_iter"]
            viewpoint = self.viewpoints[self.current_window[0]]
            for mapping_iter in range(iters_recent):
                loss_mapping = 0
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                #print("render time is " , t2.toc() , " ms")
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                
                # if cam_idx == len(current_window) - 1 and mapping_iter == 2:
                #     print("mapping step and render depth mean is " , depth.mean())
                #     print("mapping step and render depth min is " , depth.min())
                #     print("mapping step and render depth max is " , depth.max())

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                # scaling = self.gaussians.get_scaling
                # mean_scaling = scaling.mean(dim=1).view(-1, 1)
                # isotropic_loss = torch.abs(scaling[visibility_filter] - mean_scaling[visibility_filter])
                # loss_mapping += 2 * isotropic_loss.mean()
                loss_mapping.backward()


                with torch.no_grad():
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

        for mapping_iter in range(iters):
            
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):  # all view points
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                #print(self.gaussians.get_xyz.shape)
                t2 = TicToc() ; t2.tic()
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                #print("render time is " , t2.toc() , " ms")
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                
                # if cam_idx == len(current_window) - 1 and mapping_iter == 2:
                #     print("mapping step and render depth mean is " , depth.mean())
                #     print("mapping step and render depth min is " , depth.min())
                #     print("mapping step and render depth max is " , depth.max())

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                
            # if map_recent:
            #     # recent_opt_params = []
            #     # for recent_idx in self.recent_windows:  # all view points
            #     #     viewpoint = self.recent_viewpoints[recent_idx]
            #     #     if viewpoint.uid == 0:
            #     #         continue
            #     #     recent_opt_params.append(
            #     #             {
            #     #                 "params": [viewpoint.cam_rot_delta],
            #     #                 "lr": self.config["Training"]["lr"]["cam_rot_delta"]
            #     #                 * 0.5,
            #     #                 "name": "rot_{}".format(viewpoint.uid),
            #     #             }
            #     #         )
            #     #     recent_opt_params.append(
            #     #         {
            #     #             "params": [viewpoint.cam_trans_delta],
            #     #             "lr": self.config["Training"]["lr"][
            #     #                 "cam_trans_delta"
            #     #             ]
            #     #             * 0.5,
            #     #             "name": "trans_{}".format(viewpoint.uid),
            #     #         }
            #     #     )
            #     #     recent_opt_params.append(
            #     #         {
            #     #             "params": [viewpoint.exposure_a],
            #     #             "lr": 0.01,
            #     #             "name": "exposure_a_{}".format(viewpoint.uid),
            #     #         }
            #     #     )
            #     #     recent_opt_params.append(
            #     #         {
            #     #             "params": [viewpoint.exposure_b],
            #     #             "lr": 0.01,
            #     #             "name": "exposure_b_{}".format(viewpoint.uid),
            #     #         }
            #     #     )
            #     # self.recentframe_optimizers = torch.optim.Adam(recent_opt_params)

            #     for recent_idx in self.recent_windows:  # all view points
            #         # lens = len(self.recent_windows)
            #         # viewpoint = self.recent_viewpoints[lens - 1 - recent_idx]
            #         render_pkg = render(
            #             viewpoint, self.gaussians, self.pipeline_params, self.background
            #         )
            #         (
            #             image,
            #             viewspace_point_tensor,
            #             visibility_filter,
            #             radii,
            #             depth,
            #             opacity,
            #             n_touched,
            #         ) = (
            #             render_pkg["render"],
            #             render_pkg["viewspace_points"],
            #             render_pkg["visibility_filter"],
            #             render_pkg["radii"],
            #             render_pkg["depth"],
            #             render_pkg["opacity"],
            #             render_pkg["n_touched"],
            #         )
            #         render_times += 1

            #         loss_mapping += get_loss_mapping(
            #             self.config, image, depth, viewpoint, opacity
            #        )

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            
            co_visibility_filter = torch.zeros_like(visibility_filter_acm[0])
            for visibility_filter in visibility_filter_acm:
                co_visibility_filter = torch.logical_or(co_visibility_filter, visibility_filter)
            scaling = self.gaussians.get_scaling
            mean_scaling = scaling.mean(dim=1).view(-1, 1)
            isotropic_loss = torch.abs(scaling[co_visibility_filter] - mean_scaling[co_visibility_filter])
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[0]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:   #prune if monocular
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                            print("prune points")
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    #print("densify and prune")
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                # if (self.iteration_count % self.gaussian_reset) == 0 and (
                #     not update_gaussian
                # ):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                #     #self.gaussians.reset_opacity_visible(visibility_filter_acm)
                #     gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # # if map_recent:
                # #     self.recentframe_optimizers.step()
                # #     self.recentframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

                    

                # for recent_idx in self.recent_windows:  # all view points
                #     viewpoint = self.recent_viewpoints[recent_idx]
                #     if viewpoint.uid == 0:
                #         continue
                #     update_pose(viewpoint)

            # if (self.iteration_count % self.gaussian_reset) == 0 and (
            #     not update_gaussian
            # ):
            #     #slidewin + rgbd
            #     print("begin slidewin + rgbd")
            #     viewpoint_idx_stack = list(self.viewpoints.keys())
            #     #print(viewpoint_idx_stack)
            #     slide_windows = 3 ; win_overlap = 0

            #     for iteration in tqdm(range(1 , 20+1)):
            #         start_id = 0 ; 
            #         while start_id < len(viewpoint_idx_stack)-1:
            #             total_loss = 0.
            #             if iteration == 1:
            #                 print(start_id )
                        
            #             viewpoint_cam_idx = viewpoint_idx_stack[start_id]
            #             viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            #             render_pkg = render(
            #                 viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            #             )
            #             image, visibility_filter, depth, radii = (
            #                 render_pkg["render"],
            #                 render_pkg["visibility_filter"],
            #                 render_pkg["depth"],
            #                 render_pkg["radii"],
            #             )

            #             loss = get_loss_mapping_rgbd2(self.config, image, depth, viewpoint_cam, initialization=False)
            #             total_loss += loss
            #             #print(total_loss)
            #             total_loss.backward()
            #             with torch.no_grad():
            #                 self.gaussians.max_radii2D[visibility_filter] = torch.max(
            #                     self.gaussians.max_radii2D[visibility_filter],
            #                     radii[visibility_filter],
            #                 )
            #                 self.gaussians.optimizer.step()
            #                 self.gaussians.optimizer.zero_grad(set_to_none=True)
            #                 self.gaussians.update_learning_rate(iteration)

            #                 start_id += slide_windows - win_overlap  

        # if iters >= 10:
        #     print("mapping iter is " , iters , " and mapping recent is " , map_recent , "and mapping time is " , map_time.toc())
        
        #print("after mapping self.backend queue is " , view_queue(self.backend_queue))
        # if iters > 50:
        #     size = self.backend_queue.qsize()  # 获取队列的大小
        #     print("back queue size is " , size)
        #     for k in range(size):
        #         element = self.backend_queue.get()
        #         print(element)
        #         if k == size - 1:
        #             self.recent_windows = element[2]
        #             self.recent_viewpoints = element[1]
        # if iters >5:
        #     print("iter is " , iters ,  " and render times is " , render_times  ,  " and mapping time is " , (map_time.toc()) , " ms")
        return

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        #recent_frames = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))  #only change kf ids

        # for recent_idx in self.recent_windows:
        #     recent_viewpoint = self.recent_viewpoints[recent_idx]
        #     recent_frames.append((recent_idx, recent_viewpoint.R.clone(), recent_viewpoint.T.clone()))  #only change kf ids
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                #print("empty queue! mapping")
                #print("self recent windwos is " , self.recent_windows)
                self.map(self.current_window , map_recent = False)  #map current windwos for keyframes
                #print("mapping done and iters is " , self.iteration_count)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    #torch.cuda.empty_cache()
                    # self.map(self.current_window, iters = 1, prune=False)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "cur_frame":   #not key frames
                    recent_viewpoints = data[1]
                    recent_windows = data[2]
                    self.recent_viewpoints = recent_viewpoints
                    self.recent_windows = recent_windows
                    print("update recent windows and viewpoints")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    new_viewpoint = data[2]    #should consider cur visibility
                    current_window = data[3]
                    depth_map = data[4]
                    pose_update = data[5]
                    pose_graph = data[6]  #path.poses

                    #if pose_update:
                    if False:
                        assert(len(pose_graph) >= len(self.viewpoints))  #update all viewpoints poses based on pose graph
                        viewpoint_idx_stack = list(self.viewpoints.keys())
                        
                        pose_id = 0
                        for i in range(len(viewpoint_idx_stack)):
                            kf_id = viewpoint_idx_stack[i]
                            while stamp2seconds(pose_graph[pose_id]) < self.viewpoints[kf_id].timestamp:
                                pose_id += 1
                            assert(stamp2seconds(pose_graph[pose_id]) == self.viewpoints[kf_id].timestamp)

                            pose_Wbk = odom2pose(pose_graph[pose_id])
                            pose_Wck = pose_Wbk@self.T_bc
                            pose_ckW = torch.inverse(pose_Wck).float()
                            self.viewpoints[kf_id].R = pose_ckW[:3 , :3]
                            self.viewpoints[kf_id].T = pose_ckW[:3 , 3]
                            





                        
#original rand + l1loss
                        # for iteration in tqdm(range(1, 26000 + 1)):
                        #     viewpoint_idx_stack = list(self.viewpoints.keys())
                        #     viewpoint_cam_idx = viewpoint_idx_stack.pop(
                        #         random.randint(0, len(viewpoint_idx_stack) - 1)
                        #     )
                        #     viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                        #     render_pkg = render(
                        #         viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                        #     )
                        #     image, visibility_filter, radii = (
                        #         render_pkg["render"],
                        #         render_pkg["visibility_filter"],
                        #         render_pkg["radii"],
                        #     )

                        #     gt_image = viewpoint_cam.original_image.cuda()
                        #     Ll1 = l1_loss(image, gt_image)
                        #     loss = (1.0 - self.opt_params.lambda_dssim) * (
                        #         Ll1
                        #     ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        #     loss.backward()
                        #     with torch.no_grad():
                        #         self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        #             self.gaussians.max_radii2D[visibility_filter],
                        #             radii[visibility_filter],
                        #         )
                        #         self.gaussians.optimizer.step()
                        #         self.gaussians.optimizer.zero_grad(set_to_none=True)
                        #         self.gaussians.update_learning_rate(iteration)








                        #rand + rgbd
                        # for iteration in tqdm(range(1 , 26000+1)):   
                        #     viewpoint_idx_stack = list(self.viewpoints.keys())
                        #     viewpoint_cam_idx = viewpoint_idx_stack.pop(
                        #         random.randint(0, len(viewpoint_idx_stack) - 1)
                        #     )
                        #     viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                        #     render_pkg = render(
                        #         viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                        #     )
                        #     image, visibility_filter, depth, radii = (
                        #                 render_pkg["render"],
                        #                 render_pkg["visibility_filter"],
                        #                 render_pkg["depth"],
                        #                 render_pkg["radii"],
                        #             )

                        #     loss = get_loss_mapping_rgbd2(self.config, image, depth, viewpoint_cam, initialization=False)
                        #     loss.backward()
                        #     with torch.no_grad():
                        #         self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        #             self.gaussians.max_radii2D[visibility_filter],
                        #             radii[visibility_filter],
                        #         )
                        #         self.gaussians.optimizer.step()
                        #         self.gaussians.optimizer.zero_grad(set_to_none=True)
                        #         self.gaussians.update_learning_rate(iteration)


                    

                        #slidewin + l1loss
                        # slide_windows = 8 ; win_overlap = 2

                        # for iteration in tqdm(range(1 , 260+1)):
                        #     start_id = 0 ; end_id = start_id + slide_windows
                        #     while start_id < len(viewpoint_idx_stack):
                        #         total_loss = 0
                        #         for i in range(start_id , end_id):
                        #             viewpoint_cam_idx = viewpoint_idx_stack[i]
                        #             viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                        #             render_pkg = render(
                        #                 viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                        #             )
                        #             image, visibility_filter, radii = (
                        #                 render_pkg["render"],
                        #                 render_pkg["visibility_filter"],
                        #                 render_pkg["radii"],
                        #             )

                        #             gt_image = viewpoint_cam.original_image.cuda()
                        #             Ll1 = l1_loss(image, gt_image)
                        #             loss = (1.0 - self.opt_params.lambda_dssim) * (
                        #                 Ll1
                        #             ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        #             total_loss += loss
                                    
                        #         total_loss.backward()
                        #         with torch.no_grad():
                        #             self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        #                 self.gaussians.max_radii2D[visibility_filter],
                        #                 radii[visibility_filter],
                        #             )
                        #             self.gaussians.optimizer.step()
                        #             self.gaussians.optimizer.zero_grad(set_to_none=True)
                        #             self.gaussians.update_learning_rate(iteration)

                        #         start_id += slide_windows - win_overlap  ;  end_id += slide_windows - win_overlap
                        #         end_id = min(len(viewpoint_idx_stack) , end_id)




                        #slidewin + rgbd  BA refinement
                        
                        slide_windows = 8 ; win_overlap = 2
                        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")  #reset all gradients
                        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
                        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")

                        for iteration in tqdm(range(1 , 100+1)):
                            start_id = 0 ; end_id = start_id + slide_windows ; end_id = min(len(viewpoint_idx_stack) , end_id)
                            while start_id < len(viewpoint_idx_stack)-1:  # pose bianli
                                total_loss = 0.
                                opt_params = []


                                cur_visibility_filter_acm = []
                                cur_radii_acm = []
                                cur_viewspace_point_tensor_acm = []


                                if iteration > 70:   #pose update
                                    for idx in range(start_id , end_id):
                                        viewpoint_cam_idx = viewpoint_idx_stack[idx]
                                        if viewpoint_cam_idx == 0:
                                            continue
                                        viewpoint = self.viewpoints[viewpoint_cam_idx]
                                        opt_params.append(
                                            {
                                                "params": [viewpoint.cam_rot_delta],
                                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                                * 0.5,
                                                "name": "rot_{}".format(viewpoint.uid),
                                            }
                                        )
                                        opt_params.append(
                                            {
                                                "params": [viewpoint.cam_trans_delta],
                                                "lr": self.config["Training"]["lr"][
                                                    "cam_trans_delta"
                                                ]
                                                * 0.5,
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
                                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                                for i in range(start_id , end_id):
                                    viewpoint_cam_idx = viewpoint_idx_stack[i]
                                    viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                                    render_pkg = render(
                                        viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                                    )
                                    image, viewspace_points_tensor, visibility_filter, depth, radii = (
                                        render_pkg["render"],
                                        render_pkg["viewspace_points"],
                                        render_pkg["visibility_filter"],
                                        render_pkg["depth"],
                                        render_pkg["radii"],
                                    )

                                    cur_visibility_filter_acm.append(visibility_filter)
                                    cur_radii_acm.append(radii)
                                    cur_viewspace_point_tensor_acm.append(viewspace_points_tensor)
                                    loss = get_loss_mapping_rgbd2(self.config, image, depth, viewpoint_cam, initialization=False)
                                    total_loss += loss

                                cur_co_visibility_filter = torch.zeros_like(cur_visibility_filter_acm[0])
                                for cur_visibility_filter in cur_visibility_filter_acm:
                                    cur_co_visibility_filter = torch.logical_or(cur_co_visibility_filter, cur_visibility_filter)

                                scaling = self.gaussians.get_scaling
                                mean_scaling = scaling.mean(dim=1).view(-1, 1)
                                isotropic_loss = torch.abs(scaling[cur_co_visibility_filter] - mean_scaling[cur_co_visibility_filter])
                                total_loss += 10 * isotropic_loss.mean()
                                total_loss.backward()
                                with torch.no_grad():
                                    for new_idx in range(len(cur_viewspace_point_tensor_acm)):
                                        self.gaussians.max_radii2D[cur_visibility_filter_acm[new_idx]] = torch.max(
                                            self.gaussians.max_radii2D[cur_visibility_filter_acm[new_idx]],
                                            cur_radii_acm[new_idx][cur_visibility_filter_acm[new_idx]],
                                        )
                                        self.gaussians.add_densification_stats(
                                            cur_viewspace_point_tensor_acm[new_idx], cur_visibility_filter_acm[new_idx]
                                        )
                                    self.gaussians.optimizer.step()
                                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                                    self.gaussians.update_learning_rate(iteration)

                                    if iteration == 40 or iteration == 90:
                                        self.gaussians.densify_and_prune(
                                            self.opt_params.densify_grad_threshold,
                                            self.init_gaussian_th,
                                            self.init_gaussian_extent,
                                            None,
                                        )

                                    if iteration > 70:   #pose update
                                        self.keyframe_optimizers.step()
                                        self.keyframe_optimizers.zero_grad(set_to_none=True)
                                        # # if map_recent:
                                        # #     self.recentframe_optimizers.step()
                                        # #     self.recentframe_optimizers.zero_grad(set_to_none=True)
                                        # Pose update
                                        for idx in range(start_id , end_id):
                                            viewpoint_cam_idx = viewpoint_idx_stack[idx]
                                            viewpoint = self.viewpoints[viewpoint_cam_idx]
                                            if viewpoint.uid == 0:
                                                continue
                                            update_pose(viewpoint)
                                        #torch.cuda.empty_cache()

                                start_id += slide_windows - win_overlap  ;  end_id += slide_windows - win_overlap
                                end_id = min(len(viewpoint_idx_stack) , end_id)
                        #torch.cuda.empty_cache()
                        Log("Map refinement for pose graph update done")

                    self.gaussians.update_learning_rate(self.iteration_count)
                    self.viewpoints[cur_frame_idx] = new_viewpoint
                    self.current_window = current_window

                    with torch.no_grad():
                        cur_render_pkg = render(  #render once
                            new_viewpoint, self.gaussians, self.pipeline_params, self.background
                        )
                        cur_gaussian_points = self.gaussians.get_xyz[cur_render_pkg["visibility_filter"]]
                        cur_gaussian_points = cur_gaussian_points.clone().detach()


                    self.add_next_kf2(cur_frame_idx,  new_viewpoint, cur_gaussian_points , depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    # print("iter per kf is " , iter_per_kf)
                    # print("begin mapping new keyframe")
                    kf_time = TicToc() ; kf_time.tic()
                    self.map(self.current_window, iters=iter_per_kf , map_recent = True)
                    self.map(self.current_window, prune=True)
                    #torch.cuda.empty_cache()
                    # self.map(self.current_window, iters = 1, prune=False)
                    #print("keyframe mapping time is " , kf_time.toc())
                    if pose_update:
                        self.push_to_frontend("keyframe pose update")
                    else:
                        self.push_to_frontend("keyframe")
                    #self.keyframe_optimizers = None ; opt_params = None
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
