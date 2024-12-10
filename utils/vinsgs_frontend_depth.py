import time
import rospy
import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.logging_utils import TicToc
from utils.convert_stamp import stamp2seconds , TicToc , inverse_Tmatrix
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
import math
import pandas as pd
from submodules.depth_anything_v2.dpt import DepthAnythingV2


def load_poses(path , T_i_c0):
    data = pd.read_csv(path)
    data = data.to_numpy()
    frames = []
    for i in range(data.shape[0]):
        trans = data[i, 1:4]
        quat = data[i, 4:8]
        quat = quat[[1, 2, 3, 0]]  #xyzw
        
        rotation_matrix = Rotation.from_quat(quat).as_matrix()

        T_w_i = np.eye(4)
        T_w_i[:3 , :3] = rotation_matrix
        T_w_i[:3, 3] = trans
        T_w_c = T_w_i@T_i_c0

        frame = {
            "timestamp": data[i, 0]*1e-9,
            "transform_matrix": np.linalg.inv(T_w_c),  #numpy
        }

        frames.append(frame)
    
    return frames


class Vins_FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        #self.monocular = config["Training"]["monocular"]
        self.monocular = False
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

        self.width = config["Dataset"]["Calibration"]["width"]
        self.height = config["Dataset"]["Calibration"]["height"]
        self.fx = config["Dataset"]["Calibration"]["fx"]
        self.fy = config["Dataset"]["Calibration"]["fy"]
        self.cx = config["Dataset"]["Calibration"]["cx"]
        self.cy = config["Dataset"]["Calibration"]["cy"]
        self.input_folder = config["Dataset"]["dataset_path"]


        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        self.depth_model = DepthAnythingV2(**self.model_configs[self.encoder])
        self.depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.depth_model.eval()
        self.depth_model.to('cuda')

        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])
        self.dist_coeffs = np.array((config["Dataset"]["Calibration"]["k1"] , config["Dataset"]["Calibration"]["k2"] , 
                config["Dataset"]["Calibration"]["p1"] , config["Dataset"]["Calibration"]["p2"]  , config["Dataset"]["Calibration"]["k3"]))


        self.fovx = 2*math.atan(self.width/(2*self.fx)) 
        self.fovy =  2*math.atan(self.height/(2*self.fy))

        self.T_BS = np.array(config["Dataset"]["T_BS"]).reshape(4,4)
        self.dtype = torch.float32

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        self.deltatime = 1e-2


        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.width,
            H=self.height,
        ).transpose(0, 1)

        self.projection_matrix = self.projection_matrix.to(device=self.device)
        self.cur_frame_idx = 0
    
        self.traj_gt = load_poses(config["Dataset"]["traj_path"] , self.T_BS)


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]


    def dataalign(self , image_buf , keypose_buf , key_point_cloud_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or keypose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
            
                            
            if(len(measurements) > 100):
                return measurements
            # if(stamp2seconds(image_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])  or  stamp2seconds(key_point_cloud_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])):  #smaller than the first keypose
            #     return measurements
            
            # if(stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(image_buf.queue[0])  and  stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(key_point_cloud_buf.queue[0])):
            #     keypose_buf.get()
            #     continue
            
                
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ; keyposetime_first = stamp2seconds(keypose_buf.queue[0]) ; keypointcloudtime_first = stamp2seconds(key_point_cloud_buf.queue[0])
            time_array = np.array([imagetime_first , keyposetime_first , keypointcloudtime_first])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)

            if id_first != 0:
                while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf.get()
            if id_first != 1:
                while  (not keypose_buf.empty()) and  (time_first   - stamp2seconds(keypose_buf.queue[0]) > self.deltatime):  #pop the data before it
                    keypose_buf.get()
            if id_first != 2:
                while    (not key_point_cloud_buf.empty()) and  (time_first   - stamp2seconds(key_point_cloud_buf.queue[0]) > self.deltatime):  #pop the data before it
                    key_point_cloud_buf.get()

            if(image_buf.empty() or keypose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
            
            cur_image = image_buf.get()  ; cur_keypose = keypose_buf.get() ; cur_key_point_cloud = key_point_cloud_buf.get()
            
            time_cur_image = stamp2seconds(cur_image) ; time_cur_keypose = stamp2seconds(cur_keypose) ; time_cur_key_point_cloud = stamp2seconds(cur_key_point_cloud)
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_keypose) < self.deltatime) ; assert(abs(time_cur_image - time_cur_key_point_cloud) < self.deltatime)

            image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
            #print(image.shape)
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image , cur_keypose , cur_key_point_cloud , time_cur_image])
            
        return measurements
    

    def dataalign2(self , image_buf , campose_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or campose_buf.empty() ):
                return measurements
                            
            if(len(measurements) > 100):
                return measurements
            # if(stamp2seconds(image_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])  or  stamp2seconds(key_point_cloud_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])):  #smaller than the first keypose
            #     return measurements
            
            # if(stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(image_buf.queue[0])  and  stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(key_point_cloud_buf.queue[0])):
            #     keypose_buf.get()
            #     continue
            
                
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ; camposetime_first = stamp2seconds(campose_buf.queue[0])
            time_array = np.array([imagetime_first , camposetime_first])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)

            if id_first != 0:
                while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf.get()
            if id_first != 1:
                while  (not campose_buf.empty()) and  (time_first   - stamp2seconds(campose_buf.queue[0]) > self.deltatime):  #pop the data before it
                    campose_buf.get()
            if(image_buf.empty() or campose_buf.empty()):
                return measurements
            
            cur_image = image_buf.get()  ; cur_campose = campose_buf.get()
   
            
            time_cur_image = stamp2seconds(cur_image) ; time_cur_campose = stamp2seconds(cur_campose) 
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_campose) < self.deltatime) 

            image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
            #print(image.shape)
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image , cur_campose , time_cur_image])
            
        return measurements
    


    def dataalign3(self , image_buf , campose_buf , key_point_cloud_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or campose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
                            
            if(len(measurements) > 100):
                return measurements
            
            
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ; camposetime_first = stamp2seconds(campose_buf.queue[0]) ;  key_point_cloud_first = stamp2seconds(key_point_cloud_buf.queue[0])
            time_array = np.array([imagetime_first , camposetime_first , key_point_cloud_first])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)
            if id_first != 2:
                while (not key_point_cloud_buf.empty()) and  (time_first   - stamp2seconds(key_point_cloud_buf.queue[0]) > self.deltatime):  #pop the data before it
                    key_point_cloud_buf.get()
            if key_point_cloud_buf.empty():
                return measurements
            time_first = stamp2seconds(key_point_cloud_buf.queue[0])

            while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                image_buf.get()
            while  (not campose_buf.empty()) and  (time_first   - stamp2seconds(campose_buf.queue[0]) > self.deltatime):  #pop the data before it
                campose_buf.get()
            if(image_buf.empty() or  campose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
        
            cur_image = image_buf.get()  ; cur_campose = campose_buf.get() ; cur_key_point_cloud = key_point_cloud_buf.get()
   
            
            time_cur_image = stamp2seconds(cur_image) ; time_cur_campose = stamp2seconds(cur_campose)  ; time_cur_keypoint = stamp2seconds(cur_key_point_cloud)
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_campose) < self.deltatime)  ; assert(abs(time_cur_image - time_cur_keypoint) < self.deltatime) 
            image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
            image_BGR = image.copy()
            #print(image.shape)
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image , image_BGR  , cur_campose , cur_key_point_cloud  , time_cur_image])
            
        return measurements

    def get_measurements(self , image_buf , keypose_buf , key_point_cloud_buf):
        return self.dataalign( image_buf , keypose_buf , key_point_cloud_buf)

    def get_measurements2(self , image_buf , campose_buf):
        return self.dataalign2( image_buf , campose_buf)

    def get_measurements3(self , image_buf , campose_buf , key_point_cloud_buf):
        return self.dataalign3( image_buf , campose_buf , key_point_cloud_buf)


    def convert_ros_image_to_cv2(self, ros_image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
            return cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return None

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
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
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R, viewpoint.T)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        # print("prev R and T is " , prev.R , prev.T)
        #viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        # opt_params.append(
        #     {
        #         "params": [viewpoint.cam_rot_delta],
        #         "lr": self.config["Training"]["lr"]["cam_rot_delta"],
        #         "name": "rot_{}".format(viewpoint.uid),
        #     }
        # )
        # opt_params.append(
        #     {
        #         "params": [viewpoint.cam_trans_delta],
        #         "lr": self.config["Training"]["lr"]["cam_trans_delta"],
        #         "name": "trans_{}".format(viewpoint.uid),
        #     }
        # )
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
        print("begin tracking and cur gaussian size is " , self.gaussians._xyz.shape)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
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
                #converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            # if converged:
            #     print("Tracking converged and actual tracking iters is " , tracking_itr )
            #     break

        self.median_depth = get_median_depth(depth, opacity)
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
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
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

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self,measurements):
        #cur_frame_idx = 0
        
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        image_id = 0
        avg_feature_points = 1.
        while True:

            if len(measurements) == 0:
                return
            
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

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                [image, image_BGR , cur_campose , cur_key_point_cloud , timestamp] = measurements.pop(0)

                
                tracking_time = TicToc()
                tracking_time.tic()

              
                
                position = cur_campose.pose.pose.position  #R_wbk
                orientation = cur_campose.pose.pose.orientation   #T_wbk
                
                

                translation = np.array([position.x, position.y, position.z ])
                quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
                rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
                
                T_Wck = np.eye(4)
                T_Wck[ :3 ,  :3 ] = rotation_matrix ; T_Wck[ :3 , 3 ] = translation

                T_Wck = torch.from_numpy((T_Wck).astype(float)).to("cuda")
                T_ckW = inverse_Tmatrix(T_Wck)

                points_3D_ext = torch.tensor([[point.x, point.y, point.z , 1] for point in cur_key_point_cloud.points]) 
                
                depth_matrix = None
                depth_mask = None
                

                if points_3D_ext.shape[0] > 0:
                    depth_matrix = torch.zeros((1 , self.height , self.width)).cuda()
                    depth_mask = torch.zeros_like(depth_matrix, dtype=torch.bool).cuda()
                    points_3D_ext = points_3D_ext.to("cuda")

                    points_3D_ext_cam = points_3D_ext@T_ckW.T
                    points_3D_cam = points_3D_ext_cam[: , :3]  #得到在当前相机坐标系下表示的点云 N*3
                    points_3D_cam = points_3D_cam[points_3D_cam[: , 2] >= 0.01]
                    points_3D_cam = points_3D_cam[points_3D_cam[: , 2] < 30]

                    #point_depth = torch.norm(points_3D_cam , dim = -1)
                    point_depth2 = points_3D_cam[: , 2]  #Z depth
                  
                    avg_feature_points = point_depth2.mean().cpu()


                    points_2D_cam = points_3D_cam@(torch.from_numpy(self.K.T).to("cuda")).to(torch.float) 
                    points_2D = points_2D_cam[:, :2] / (points_2D_cam[:, 2].reshape(-1, 1))  # Nx2
                    points_2D = points_2D.int()
                    #print("max of point2D is " , points_2D[: , 0].max() , points_2D[: , 1].max())
                    valid_mask1 = torch.logical_and(points_2D[:, 0] >= 0 ,(points_2D[:,0] <= self.width - 1))
                    valid_mask2 = torch.logical_and(points_2D[:,1] >= 0 , (points_2D[:,1] <= self.height - 1))
                    valid_mask = torch.logical_and(valid_mask1 , valid_mask2)

                    points_2D = points_2D[valid_mask]
                    point_depth2 = point_depth2[valid_mask]
                    points_2D = points_2D.long()
                    print("avg of feature point depth is " , point_depth2.mean())
                    print("min of feature point depth is " , point_depth2.min())
                    print("max of feature point depth is " , point_depth2.max())
                    # # Populate gt_depth_matrix and depth_mask
                    depth_matrix[0, points_2D[:, 1], points_2D[:, 0]] = point_depth2
                    depth_mask[0, points_2D[:, 1], points_2D[:, 0]] = True

                
                
                #print(T_Wck)
                infer_time1 = time.time()
                gt_depth_map = self.depth_model.infer_image(image_BGR) # HxW raw depth map in numpy
                # gt_depth_map[gt_depth_map < 0.04] = 1e10
                # gt_depth_map = 47./gt_depth_map

                gt_depth_map =  gt_depth_map.max() + gt_depth_map.min() - gt_depth_map
                gt_depth_map[gt_depth_map < 0.04] = 0.04 
                # normalized_depth_any = cv2.normalize(gt_depth_map, None, 0, 255, cv2.NORM_MINMAX)

                # # 转换为8位图像
                # depth_8bit_any = np.uint8(normalized_depth_any)

                # # 使用颜色映射进行可视化
                # colored_depth_any = cv2.applyColorMap(depth_8bit_any, cv2.COLORMAP_JET)
                # cv2.imshow("pred depth" , colored_depth_any)
                # cv2.waitKey(0)
                # print("infer_time is " , time.time() - infer_time1)
                #print(avg_feature_points.device)  ;  print(gt_depth_map[points_2D[:, 1].cpu() , points_2D[:, 0].cpu() ].mean())
                #if points_3D_ext.shape[0] > 0:
                gt_feature_depth = gt_depth_map[points_2D[:, 1].cpu() , points_2D[:, 0].cpu() ]
                
                stereo_any = avg_feature_points/gt_feature_depth[gt_feature_depth > 0.01].mean()
                print("stereo/any scale is " , stereo_any)
                depth_scale_any = gt_depth_map * float(stereo_any)  # chu shi shendu tu 
                print("avg of depth_scale_any is " , depth_scale_any.mean())
                print("min of depth_scale_any is " , depth_scale_any.min())
                print("max of depth_scale_any is " , depth_scale_any.max())

                while (timestamp - self.traj_gt[image_id]["timestamp"]) > self.deltatime:
                    image_id += 1
                print("find image")
                if image_id  >= len(self.traj_gt):
                    return
                viewpoint = Camera(uid = self.cur_frame_idx, color = image, depth = depth_scale_any, gt_T = torch.from_numpy(self.traj_gt[image_id]["transform_matrix"]).to("cuda"), projection_matrix = self.projection_matrix
                    , fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy , fovx=self.fovx, fovy=self.fovy, image_height=self.height, image_width=self.width , trian_depthmap=depth_matrix , trian_depthmask=depth_mask  , init_T = T_ckW , 
                    timestamp=timestamp)
                viewpoint.compute_grad_mask(self.config)

                #print("Frontend !!! Prepare viewpoint and 3D point cost " , tracking_time.toc()  ,"ms")

                self.cameras[self.cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(self.cur_frame_idx, viewpoint)
                    self.current_window.append(self.cur_frame_idx)
                    self.cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(self.cur_frame_idx, viewpoint)

               # print("Frontend !!! Tracking and refining exposure cost " , tracking_time.toc()  ,"ms")

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                #print("Frontend !!! Push to UI cost " , tracking_time.toc()  ,"ms")

                if self.requested_keyframe > 0:
                    self.cleanup(self.cur_frame_idx)
                    self.cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (self.cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    self.cur_frame_idx,
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

                #print("Frontend !!! Judge Kf cost " , tracking_time.toc()  ,"ms")

                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        self.cur_frame_idx,
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
                        self.cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        self.cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    # print("Frontend !!! Create Kf cost " , tracking_time.toc()  ,"ms")
                    # print("current window is " , [cur_win for cur_win in self.current_window])
                    # print("current gaussian size  is " , self.gaussians._xyz.shape)
                else:
                    self.cleanup(self.cur_frame_idx)
                self.cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", self.cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        self.cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
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
