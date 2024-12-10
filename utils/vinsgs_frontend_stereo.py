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
from utils.slam_utils import get_loss_tracking, get_median_depth , get_loss_mapping
from utils.logging_utils import TicToc
from utils.convert_stamp import stamp2seconds , TicToc
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
import math
import pandas as pd
import glob


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


        self.recent_window = []
        self.recent_viewpoints = {}

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
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
        self.input_folder = config["Dataset"]["dataset_path"]

        self.width = config["Dataset"]["Calibration"]["width"]
        self.height = config["Dataset"]["Calibration"]["height"]

        cam0raw = config["Dataset"]["Calibration"]["cam0"]["raw"]
        cam1raw = config["Dataset"]["Calibration"]["cam1"]["raw"]


        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.K_raw = np.array([[self.fx_raw, 0, self.cx_raw],
                           [0, self.fy_raw, self.cy_raw],
                           [0, 0, 1]])
        

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.K_raw_r = np.array([[self.fx_raw_r, 0, self.cx_raw_r],
                           [0, self.fy_raw_r, self.cy_raw_r],
                           [0, 0, 1]])


        camopt = config["Dataset"]["Calibration"]["cam0"]["opt"]
        self.fx = camopt["fx"]
        self.fy = camopt["fy"]
        self.cx = camopt["cx"]
        self.cy = camopt["cy"]
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])
        



        self.dist_coeffs = np.array((cam0raw["k1"] , cam0raw["k2"] , 
               cam0raw["p1"] , cam0raw["p2"]  , cam0raw["k3"]))

        self.dist_coeffs_r = np.array((cam1raw["k1"] , cam1raw["k2"] , 
                cam1raw["p1"] , cam1raw["p2"]  , cam1raw["k3"]))

        self.T_BS = np.array(config["Dataset"]["Calibration"]["cam0"]["T_BS"]).reshape(4,4)
        self.T_BS_r = np.array(config["Dataset"]["Calibration"]["cam1"]["T_BS"]).reshape(4,4)
        T_SS_r = np.linalg.inv(self.T_BS)@self.T_BS_r
        print("T S S_r is " , T_SS_r)
        self.baseline_fx =  np.linalg.norm(T_SS_r[:3 , 3])*self.fx
        print("baseline times fx is " , self.baseline_fx)

        self.fovx = 2*math.atan(self.width/(2*self.fx)) 
        self.fovy =  2*math.atan(self.height/(2*self.fy))

        self.dtype = torch.float32

        self.Rmat = np.array(config["Dataset"]["Calibration"]["cam0"]["R"]["data"]).reshape(3, 3)
        self.Rmat_r = np.array(config["Dataset"]["Calibration"]["cam1"]["R"]["data"]).reshape(3, 3)


        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
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

        self.prev_pose = None ; self.prev_idx = None



        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )

        self.gtimage_timestamps = [ float(color_path.split("/")[-1].split(".")[0])/1e9 for color_path in self.color_paths]
        self.gttimestamp_id = 0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        #self.tracking_itr_num = 20
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
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image , cur_keypose , cur_key_point_cloud , time_cur_image])
            
        return measurements
    

    # def dataalign2(self , image_buf , image_buf2  , campose_buf): #align the data k1 k2 p1 p2 k3
        
    #     measurements = []
        
    #     while(True):
    #         if(image_buf.empty() or image_buf2.empty() or campose_buf.empty() ):
    #             return measurements
    #         if(len(measurements) > 100):
    #             return measurements
                
    #         #print("image buf size is " , image_buf.qsize() , "and image buf 2 size is " , image_buf2.qsize() , " and campose buf size is " , campose_buf.qsize())
    #         imagetime_first =  stamp2seconds(image_buf.queue[0]) ;   imagetime_first2 =  stamp2seconds(image_buf2.queue[0])  ;  camposetime_first = stamp2seconds(campose_buf.queue[0])
    #         time_array = np.array([imagetime_first , imagetime_first2 , camposetime_first])
    #         id_first = np.argmax(time_array)
    #         time_first = np.max(time_array)
    #         if id_first != 2:
    #             while (not campose_buf.empty()) and  (time_first   - stamp2seconds(campose_buf.queue[0]) > self.deltatime):  #pop the data before it
    #                 campose_buf.get()
    #         if campose_buf.empty():
    #             return measurements
    #         time_first = stamp2seconds(campose_buf.queue[0])

    #         while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
    #             image_buf.get()
    #         while  (not image_buf2.empty()) and  (time_first   - stamp2seconds(image_buf2.queue[0]) > self.deltatime):  #pop the data before it
    #             image_buf2.get()
    #         if(image_buf.empty() or  image_buf2.empty() or campose_buf.empty()):
    #             return measurements
            
    #         cur_image = image_buf.get()  ;   cur_image2 = image_buf2.get()   ;  cur_campose = campose_buf.get()
   
            
    #         time_cur_image = stamp2seconds(cur_image) ;   time_cur_image2 = stamp2seconds(cur_image2)  ;  time_cur_campose = stamp2seconds(cur_campose) 
    #         #print("image time is " , time_cur_image , "image2 time is " , time_cur_image2 , "cam pose  time is " , time_cur_campose)
    #         assert(abs(time_cur_image - time_cur_campose) < self.deltatime)  ; assert(abs(time_cur_image - time_cur_image2) < self.deltatime) 
    #         image = self.convert_ros_image_to_cv2(cur_image)   ;   image_r = self.convert_ros_image_to_cv2(cur_image2) 
           
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ; image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY )
    #         image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
    #         image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
    #         #cv2.imshow("vinsgs image_left" , image) ;  
    #         stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
    #         stereo.setUniquenessRatio(40)
    #         disparity = stereo.compute(image, image_r) / 16.0
    #         disparity[disparity == 0] = 1e10
    #         depth = self.baseline_fx / (
    #             disparity
    #         )  ## Following ORB-SLAM2 config, baseline*fx
    #         depth[depth < 0] = 0
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #         image = (
    #             torch.from_numpy(image / 255.0)
    #             .clamp(0.0, 1.0)
    #             .permute(2, 0, 1)
    #             .to(device=self.device, dtype=self.dtype)
    #         )
    #         measurements.append([image ,  depth  , cur_campose  , time_cur_image])
            
    #     return measurements


    def dataalign2(self , campose_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):

            if(campose_buf.empty() ):
                return measurements

            if(len(measurements) > 100):
                return measurements
                
            #print("image buf size is " , image_buf.qsize() , "and image buf 2 size is " , image_buf2.qsize() , " and campose buf size is " , campose_buf.qsize())
            camposetime_first = stamp2seconds(campose_buf.queue[0])
            while camposetime_first - self.gtimage_timestamps[self.gttimestamp_id] > self.deltatime:
                self.gttimestamp_id += 1
            cur_campose = campose_buf.get()
            assert(abs(camposetime_first - self.gtimage_timestamps[self.gttimestamp_id]) < self.deltatime)
            image = cv2.imread(self.color_paths[self.gttimestamp_id])
            image_r = cv2.imread(self.color_paths_r[self.gttimestamp_id])
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ; image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY )
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
            #cv2.imshow("vinsgs image_left" , image) ;  
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(image, image_r) / 16.0
            disparity[disparity == 0] = 1e10
            depth = self.baseline_fx / (
                disparity
            )  ## Following ORB-SLAM2 config, baseline*fx
            depth[depth < 0] = 0
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image ,  depth  , cur_campose  , camposetime_first])
            
        return measurements
    

    def dataalign3(self , image_buf , image_buf2): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or image_buf2.empty() ):
                return measurements
                
            if(len(measurements) > 100):
                return measurements
            #print("image buf size is " , image_buf.qsize() , "and image buf 2 size is " , image_buf2.qsize() )
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ;   imagetime_first2 =  stamp2seconds(image_buf2.queue[0]) 
            time_array = np.array([imagetime_first , imagetime_first2])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)

            if id_first != 0:
                while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf.get()
            if id_first != 1:
                while  (not image_buf2.empty()) and  (time_first   - stamp2seconds(image_buf2.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf2.get()
            
            cur_image = image_buf.get()  ;   cur_image2 = image_buf2.get() 
   
            
            time_cur_image = stamp2seconds(cur_image) ;   time_cur_image2 = stamp2seconds(cur_image2)  
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_image2) < self.deltatime) 


            image = self.convert_ros_image_to_cv2(cur_image)   ;   image_r = self.convert_ros_image_to_cv2(cur_image2) 
           
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ; image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY )
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
            #cv2.imshow("vinsgs image_left" , image) ;  
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(image, image_r) / 16.0
            disparity[disparity == 0] = 1e10
            depth = self.baseline_fx / (
                disparity
            )  ## Following ORB-SLAM2 config, baseline*fx
            depth[depth < 0] = 0
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) #RGB
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )

            # normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

            # # 转换为8位图像
            # depth_8bit = np.uint8(normalized_depth)

            # # 使用颜色映射进行可视化
            # colored_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            # cv2.imshow("vinsgs depth", colored_depth)  ;   cv2.imshow("vinsgs image_right",image_r) ; cv2.waitKey(0)
            # cv2.destroyAllWindows()
            measurements.append([image ,  depth , time_cur_image ])
            
        return measurements

    def dataalign4(self , image_buf , image_buf2  , campose_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or image_buf2.empty()):
                return measurements
            if(len(measurements) > 100):
                return measurements
                
            #print("image buf size is " , image_buf.qsize() , "and image buf 2 size is " , image_buf2.qsize() , " and campose buf size is " , campose_buf.qsize())
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ;   imagetime_first2 =  stamp2seconds(image_buf2.queue[0])  
            time_array = np.array([imagetime_first , imagetime_first2])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)


            if id_first != 0:
                while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf.get()

            if id_first != 1:
                while  (not image_buf2.empty()) and  (time_first   - stamp2seconds(image_buf2.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf2.get()

            while (not campose_buf.empty()) and  (time_first   - stamp2seconds(campose_buf.queue[0]) > self.deltatime):  #pop the data before it
                campose_buf.get()
            
            
            if(image_buf.empty() or  image_buf2.empty()):
                return measurements
            
            cur_image = image_buf.get()  ;   cur_image2 = image_buf2.get()   ;  cur_campose = None

            if (not campose_buf.empty()) and (stamp2seconds(campose_buf.queue[0]) == time_first):  #have cam pose
                cur_campose = campose_buf.get()
            
            time_cur_image = stamp2seconds(cur_image) ;   time_cur_image2 = stamp2seconds(cur_image2) 
            
            if cur_campose is not None:
                time_cur_campose = stamp2seconds(cur_campose) 
                #print("image time is " , time_cur_image , "image2 time is " , time_cur_image2 , "cam pose  time is " , time_cur_campose)
                assert(abs(time_cur_image - time_cur_campose) < self.deltatime)  
            assert(abs(time_cur_image - time_cur_image2) < self.deltatime) 

            image = self.convert_ros_image_to_cv2(cur_image)   ;   image_r = self.convert_ros_image_to_cv2(cur_image2) 
           
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ; image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY )
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
            #cv2.imshow("vinsgs image_left" , image) ;  
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(image, image_r) / 16.0
            disparity[disparity == 0] = 1e10
            depth = self.baseline_fx / (
                disparity
            )  ## Following ORB-SLAM2 config, baseline*fx
            depth[depth < 0] = 0
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            measurements.append([image ,  depth  , cur_campose  , time_cur_image])
            
        return measurements

    def get_measurements(self , image_buf , keypose_buf , key_point_cloud_buf):
        return self.dataalign( image_buf , keypose_buf , key_point_cloud_buf)

    def get_measurements2(self , campose_buf):
        return self.dataalign2( campose_buf)

    def get_measurements3(self , image_buf , image_buf2):
        return self.dataalign3( image_buf , image_buf2)
    
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
        # # print("prev R and T is " , prev.R , prev.T)
        viewpoint.update_RT(prev.R, prev.T)
       
        

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

    
        print("begin tracking and cur gaussian size is " , self.gaussians._xyz.shape)
        for tracking_itr in range(self.tracking_itr_num):
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
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
             
           
            loss_tracking = get_loss_tracking( #only update cur viewpoints
                self.config, image, depth, opacity, viewpoint
            )

            loss_tracking.backward()

            with torch.no_grad():

                # self.gaussians.max_radii2D[visibility_filter] = torch.max(
                #     self.gaussians.max_radii2D[visibility_filter],
                #     radii[visibility_filter],
                # )
                
                # self.gaussians.add_densification_stats(
                #     viewspace_point_tensor, visibility_filter
                # )
                # self.gaussians.optimizer.step()
                # self.gaussians.optimizer.zero_grad(set_to_none=True)
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                #self.gaussians.update_learning_rate(self.iteration_count)
                converged = update_pose(viewpoint)

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
            if converged:
                print("Tracking converged and actual tracking iters is " , tracking_itr )
                break
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
    



    def tracking2(self, cur_frame_idx, viewpoint):
        #prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        # # print("prev R and T is " , prev.R , prev.T)
        #sviewpoint.update_RT(prev.R, prev.T)
       
        #self.keyframe_optimizers = torch.optim.Adam(opt_params)

        # opt_params = []
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
        # opt_params.append(
        #     {
        #         "params": [viewpoint.exposure_a],
        #         "lr": 0.01,
        #         "name": "exposure_a_{}".format(viewpoint.uid),
        #     }
        # )
        # opt_params.append(
        #     {
        #         "params": [viewpoint.exposure_b],
        #         "lr": 0.01,
        #         "name": "exposure_b_{}".format(viewpoint.uid),
        #     }
        # )

        # pose_optimizer = torch.optim.Adam(opt_params)

    
        # print("begin tracking and cur gaussian size is " , self.gaussians._xyz.shape)

        time1 = TicToc() ;  time1.tic()
        for tracking_itr in range(self.tracking_itr_num//10):
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
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
             
           
            loss_tracking = get_loss_tracking( #only update cur viewpoints
                self.config, image, depth, opacity, viewpoint
            )

            loss_tracking.backward()

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
                #self.gaussians.update_learning_rate(self.iteration_count)
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
        
        self.median_depth = get_median_depth(depth, opacity)
        duration = time1.toc()
        time.sleep(max(0.4 - duration*1e-3 , 0.0))
        return render_pkg
    

    def tracking3(self, cur_frame_idx, viewpoint):

        time1 = TicToc() ;  time1.tic()
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
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
             
        self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
        
        self.median_depth = get_median_depth(depth, opacity)
        duration = time1.toc()
        time.sleep(max(0.5 - duration*1e-3 , 0.0))
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
        self.backend_queue.put(msg)   #put keyframe msg
        self.requested_keyframe += 1

    def request_curframe(self, viewpoints , recent_windows):
        msg = ["cur_frame", viewpoints, recent_windows]
        self.backend_queue.put(msg)   #put keyframe msg

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]  #put mapping msg
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        #recent_frames = data[4]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

        # for recent_id, recent_R, recent_T in recent_frames:
        #     if recent_id not in self.recent_viewpoints:
        #         continue
        #     self.recent_viewpoints[recent_id].update_RT(recent_R.clone(), recent_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self,measurements):
        #cur_frame_idx = 0
        
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        image_id = 0
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

            if self.frontend_queue.empty():  #no queue
                tracking_time = TicToc()
                tracking_time.tic()
                tic.record()

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                # if self.single_thread and self.requested_keyframe > 0:
                #     time.sleep(0.01)
                #     continue

                if self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                [image, cur_depth , cur_campose , timestamp] = measurements.pop(0)
                if self.reset and cur_campose is None:
                    continue
                print("cur frame idx is " , self.cur_frame_idx)
                print("cur timestamp is " , timestamp)
                # [image, cur_depth] = measurements.pop(0)
                # print("cur frame idx is " , self.cur_frame_idx)
                
                tracking_time = TicToc()
                tracking_time.tic()

                T_ckW = None

                if cur_campose is not None: 
                    position = cur_campose.pose.pose.position  #R_wbk
                    orientation = cur_campose.pose.pose.orientation   #T_wbk
                    
                    

                    translation = np.array([position.x, position.y, position.z])
                    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
                    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
                    
                    T_Wck = np.eye(4)
                    T_Wck[ :3 ,  :3 ] = rotation_matrix ; T_Wck[ :3 , 3 ] = translation

                    T_Wck = torch.from_numpy((T_Wck).astype(float)).to("cuda")
                    T_ckW = torch.inverse(T_Wck).float()
                
                
                #print(T_Wck)
                refine_T_ckW = None
                if self.reset:  #start
                    refine_T_ckW = T_ckW
                    self.prev_pose = T_ckW if T_ckW is not None else torch.eye(4).to("cuda").float(); self.prev_idx = self.cur_frame_idx
                else:
                    if T_ckW is not None:
                        T_ckcj = T_ckW @ torch.inverse(self.prev_pose)
                        T_cjW = torch.eye(4).to("cuda").float() ; T_cjW[0:3 , 0:3] = self.cameras[self.prev_idx].R ; T_cjW[0:3 , 3] = self.cameras[self.prev_idx].T
                        refine_T_ckW = T_ckcj@T_cjW
                        self.prev_pose = T_ckW.float() ; self.prev_idx = self.cur_frame_idx

                
                while image_id < len(self.traj_gt) and (timestamp - self.traj_gt[image_id]["timestamp"]) > self.deltatime:
                    image_id += 1
                #print("cur image id is " , image_id) ; print("cur timestamp is " , timestamp , "and gt timestamp is " , self.traj_gt[image_id]["timestamp"])
                if image_id  >= len(self.traj_gt):
                    return
                #T_ckW = torch.from_numpy(self.traj_gt[image_id]["transform_matrix"]).to("cuda")
                viewpoint = Camera(uid = self.cur_frame_idx, color = image, depth = cur_depth, gt_T = torch.from_numpy(self.traj_gt[image_id]["transform_matrix"]).to("cuda"), projection_matrix = self.projection_matrix  #RGB
                    , fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy , fovx=self.fovx, fovy=self.fovy, image_height=self.height, image_width=self.width , init_T =  T_ckW,
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
                print("camera 0 pose is " ,self.cameras[0].R , self.cameras[0].T)
                if T_ckW is not None:
                    render_pkg = self.tracking3(self.cur_frame_idx, viewpoint) #update prev pose and gaussians
                else:
                    render_pkg = self.tracking(self.cur_frame_idx, viewpoint)

                #render_pkg = self.tracking(self.cur_frame_idx, viewpoint)
                print("Frontend !!! Tracking and refining exposure cost " , tracking_time.toc()  ,"ms")

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

                # if self.requested_keyframe > 0:
                #     print("requesting key frame and cur requested_keyframe num is " , self.requested_keyframe)
                #     self.recent_viewpoints[self.cur_frame_idx] = viewpoint.copy()
                #     self.recent_window.append(self.cur_frame_idx)
                #     if(len(self.recent_window) > self.config["Training"]["window_size"]):
                #         pop_idx = self.recent_window.pop(0)
                #         self.recent_viewpoints.pop(pop_idx)
                #     self.cleanup(self.cur_frame_idx)
                #     self.cur_frame_idx += 1
                #     self.request_curframe(self.recent_viewpoints,self.recent_window) 
                #     continue

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
                print("last frame idx is " , last_keyframe_idx)
                print("self current windows is " , self.current_window)
                if create_kf:   #update pose according to the keyframe 
                    print("create_kf !!")
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



                    # opt_params = []
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
                    # opt_params.append(
                    #     {
                    #         "params": [viewpoint.exposure_a],
                    #         "lr": 0.01,
                    #         "name": "exposure_a_{}".format(viewpoint.uid),
                    #     }
                    # )
                    # opt_params.append(
                    #     {
                    #         "params": [viewpoint.exposure_b],
                    #         "lr": 0.01,
                    #         "name": "exposure_b_{}".format(viewpoint.uid),
                    #     }
                    # )
                    # print("viewpoint pose before pose update is " , viewpoint.R , viewpoint.T)
                    # prev_T = torch.eye(4).to("cuda") ; prev_T[ :3 , :3] = viewpoint.R ; prev_T[ :3 , 3] = viewpoint.T
                    # pose_optimizer = torch.optim.Adam(opt_params)
                    
                    # for tracking_itr in range(20):
                    #     render_pkg = render(
                    #         viewpoint, self.gaussians, self.pipeline_params, self.background
                    #     )
                    #     (
                    #         image,
                    #         viewspace_point_tensor,
                    #         visibility_filter,
                    #         radii,  #3 std gaussian in the screen
                    #         depth,
                    #         opacity,
                    #     ) = (
                    #         render_pkg["render"],
                    #         render_pkg["viewspace_points"],
                    #         render_pkg["visibility_filter"],
                    #         render_pkg["radii"],
                    #         render_pkg["depth"],
                    #         render_pkg["opacity"],
                    #     )
                        
                    
                    #     loss_tracking = get_loss_tracking( #only update cur viewpoints
                    #         self.config, image, depth, opacity, viewpoint
                    #     )

                    #     loss_tracking.backward()

                    #     with torch.no_grad():
                    #         pose_optimizer.step()
                    #         pose_optimizer.zero_grad()
                    #         converged = update_pose(viewpoint)

                        
                    #     # if converged:
                    #     #     print("Tracking converged and actual tracking iters is " , tracking_itr )
                    #     #     break
                    # self.median_depth = get_median_depth(depth, opacity)

                    # print("viewpoint pose after pose update is " , viewpoint.R , viewpoint.T)
                    # cur_T = torch.eye(4).to("cuda") ; cur_T[ :3 , :3] = viewpoint.R ; cur_T[ :3 , 3] = viewpoint.T
                    # update_T = torch.inverse(prev_T) @ cur_T
                    # print("pose update R and trans is  " , update_T[:3 , :3] , update_T[:3 , 3])
                    # print("cam exposure is " , viewpoint.exposure_a , viewpoint.exposure_b) 


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
                    # self.recent_viewpoints[self.cur_frame_idx] = viewpoint.copy()
                    # self.recent_window.append(self.cur_frame_idx)
                    # if(len(self.recent_window) > self.config["Training"]["window_size"]):
                    #     pop_idx = self.recent_window.pop(0)
                    #     recent_cur_viewpoint = self.recent_viewpoints.pop(pop_idx)
                    #     recent_cur_viewpoint.original_image = None 
                    #     recent_cur_viewpoint.depth = None
                    #     recent_cur_viewpoint.grad_mask = None

                    #     recent_cur_viewpoint.cam_rot_delta = None
                    #     recent_cur_viewpoint.cam_trans_delta = None

                    #     recent_cur_viewpoint.exposure_a = None
                    #     recent_cur_viewpoint.exposure_b = None
                    self.cleanup(self.cur_frame_idx)
                    #self.request_curframe(self.recent_viewpoints,self.recent_window) 
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
                    self.sync_backend(data)  #result after optimization
                    self.requested_keyframe -= 1   #no request for keyframe

                elif data[0] == "init":
                    self.sync_backend(data)  #init
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
