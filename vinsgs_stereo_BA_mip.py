#!/home/wkx123/anaconda3/envs/splatam/bin/python
# -*- coding: utf-8 -*-
import rospy
import torch
from nav_msgs.msg import Odometry , Path
from std_msgs.msg import Bool , Header
from sensor_msgs.msg import PointCloud , Image
from visualization_msgs.msg import Marker, MarkerArray
from queue import Queue
import sys
from threading import Lock, Condition, Event
from gaussian_splatting.scene.gaussian_model_mip import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui_mip
from utils.config_utils import load_config
import threading
from argparse import ArgumentParser
from datetime import datetime
import yaml
import torch.multiprocessing as mp
import os
from munch import munchify
from utils.multiprocessing_utils import FakeQueue
from utils.vinsgs_backend_BA_mip import BackEnd
from utils.vinsgs_frontend_stereo_BA_mip import Vins_FrontEnd
from gaussian_splatting.gaussian_renderer_mip import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering_vings_mip, save_gaussians
from utils.logging_utils import TicToc
import time
import wandb
from utils.pose_utils import update_pose
import glob
import torch
import cv2
import torch.nn as nn




last_imu_t = 0.0

# image_buf = Queue()
# image_buf2 = Queue()
imu_odom_buf = Queue()
point_cloud_buf = Queue()
margin_point_cloud_buf = Queue()
campose_buf = Queue()
keypose_buf = Queue()   #camera的keyposes   
odometry_buf = Queue()
key_point_cloud_buf = Queue()
key_point_cloud_incre_buf = Queue()
pose_graph_buf = Queue()
pose_update_buf = Queue()


m_buf = Lock()
con = Condition(m_buf)

# def image_callback(data):
#     with con:
#         image_buf.put(data)
#         con.notify()  

# def image_callback2(data):
#     with con:
#         image_buf2.put(data)
#         con.notify()  

# def imu_odom_callback(data):
#     with con:
#         imu_odom_buf.put(data)
#         con.notify()  

# def point_cloud_callback(data):
#     with con:
#         point_cloud_buf.put(data)
#         con.notify()  

# def margin_cloud_callback(data):
#     with con:
#         margin_point_cloud_buf.put(data)
#         con.notify()  

# def campose_callback(data):
#     with con:
#         campose_buf.put(data)
#         con.notify()  

# def odometry_callback(data):
#     with con:
#         odometry_buf.put(data)
#         con.notify() 

# def keypose_callback(data):
#     with con:
#         keypose_buf.put(data)
#         con.notify()
        
# def key_point_cloud_callback(data):
#     with con:
#         key_point_cloud_buf.put(data)
#         con.notify()

# def pose_graph_callback(data):
#     with con:
#         pose_graph_buf.put(data)
#         con.notify()

# def pose_update_callback(data):
#     with con:
#         pose_update_buf.put(data)
#         con.notify()




def imu_odom_callback(data):
    #with con:
        imu_odom_buf.put(data)
        con.notify()  

def point_cloud_callback(data):
    #with con:
        point_cloud_buf.put(data)
        con.notify()  

def margin_cloud_callback(data):
    #with con:
        margin_point_cloud_buf.put(data)
        con.notify()  

def campose_callback(data):
    #with con:
        campose_buf.put(data)
        #con.notify()  

def odometry_callback(data):
    #with con:
        odometry_buf.put(data)
        #con.notify() 

def keypose_callback(data):
    #with con:
        keypose_buf.put(data)
       # con.notify()
        
def key_point_cloud_callback(data):
   # with con:
        key_point_cloud_buf.put(data)
      #  con.notify()

def pose_graph_callback(data):
   # with con:
        pose_graph_buf.put(data)
     #   con.notify()

def pose_update_callback(data):
    #with con:
        pose_update_buf.put(data)
       # con.notify()
# def key_point_cloud_incre_callback(data):
#     with con:
#         key_point_cloud_incre_buf.put(data)
#         con.notify()


class vinsgs:
    def __init__(self, config, save_dir=None):
        
        self.shutdown_event = Event()
        self.config = config
        self.save_dir = save_dir
        self.timeout = 5.0
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        
        self.input_folder = config["Dataset"]["dataset_path"]
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = Vins_FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        self.map1x, self.map1y = self.frontend.map1x , self.frontend.map1y

        self.kernel_size = self.config["Training"]["kernel_size"]

        
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = self.q_main2vis
        self.frontend.q_vis2main = self.q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.T_bc = self.frontend.T_bc

        self.frontend_queue = frontend_queue
        self.backend_queue = backend_queue

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )



    def process(self) :
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            backend_process = mp.Process(target=self.backend.run)
            if self.use_gui:
                gui_process = mp.Process(target=slam_gui_mip.run, args=(self.params_gui,))
                gui_process.start()
                time.sleep(1)

            backend_process.start()  ; started = False
            while not rospy.is_shutdown():
                #with con:
                    #time1 = TicToc()
                measurements = self.frontend.get_measurements2(pose_graph_buf , pose_update_buf , odometry_buf , keypose_buf)
                    #print("process msg time is " , time1.toc())
                if measurements:
                    self.frontend.run(measurements)
                    last_message_time = time.time()
                    started = True
                else:
                    # Check if the timeout period has been exceeded
                    time.sleep(0.01)
                    if not started:
                        continue
                    if time.time() - last_message_time > self.timeout:
                        print("Timeout exceeded, breaking the loop.")
                        break
            
            self.backend_queue.put(["pause"])
            end.record()
            torch.cuda.synchronize()

            N_frames = len(self.frontend.cameras)
            FPS = N_frames / (start.elapsed_time(end) * 0.001)
            Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
            Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

            if self.eval_rendering:
                print("eval_rendering!!!!")
                self.gaussians = self.frontend.gaussians
                kf_indices = self.frontend.kf_indices
                ATE = eval_ate(
                    self.frontend.cameras,
                    self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )
                print("len of cam is " , len(self.frontend.cameras))
                #print("cam dict is " , self.frontend.cameras.keys())
                rendering_result = eval_rendering_vings_mip(
                    self.frontend.cameras,
                    self.input_folder,
                    self.gaussians,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    map1x = self.map1x,
                    map1y = self.map1y,
                    iteration="before_opt",
                    kernel_size=self.kernel_size
                )
                columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
                metrics_table = wandb.Table(columns=columns)
                metrics_table.add_data(
                    "Before",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )

                # re-used the frontend queue to retrive the gaussians from the backend.
                while not self.frontend_queue.empty():
                    self.frontend_queue.get()
                self.backend_queue.put(["color_refinement"])
                while True:
                    if self.frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = self.frontend_queue.get()
                    if data[0] == "sync_backend" and self.frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

                
                rendering_result = eval_rendering_vings_mip(
                    self.frontend.cameras,
                    self.input_folder,
                    self.gaussians,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    map1x = self.map1x,
                    map1y = self.map1y,
                    iteration="after_opt",
                    kernel_size=self.kernel_size
                )

                

                    
                color_paths = sorted(
                    glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
                )

                gtimage_timestamps = [ float(color_path.split("/")[-1].split(".")[0])/1e9 for color_path in color_paths]




                

                total_frame = len(self.frontend.cameras) - len(self.frontend.kf_indices)
                gttimestamp_id = 0

                for i, viewpoint in self.frontend.cameras.items():
                    print(i)
                    if (i in self.frontend.kf_indices) :
                        continue
                    pose_opt_params = []
                    
                    viewpoint.cam_rot_delta = nn.Parameter(
                        torch.zeros(3, requires_grad=True, device="cuda")
                    )
                    viewpoint.cam_trans_delta = nn.Parameter(
                        torch.zeros(3, requires_grad=True, device="cuda")
                    )
                    pose_opt_params.append(
                            {
                                "params": [viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                * 0.5,
                                "name": "rot_{}".format(viewpoint.uid),
                            }
                        )
                    pose_opt_params.append(
                            {
                                "params": [viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"][
                                    "cam_trans_delta"
                                ]
                                * 0.5,
                                "name": "trans_{}".format(viewpoint.uid),
                            }
                        )





                    while viewpoint.timestamp - gtimage_timestamps[gttimestamp_id] > 1e-3:
                        gttimestamp_id += 1
                    assert(abs(viewpoint.timestamp - gtimage_timestamps[gttimestamp_id]) < 1e-3)
                    gt_image = cv2.imread(color_paths[gttimestamp_id])
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)  # 转换颜色格式为 RGB
                    gt_image = cv2.remap(gt_image, self.frontend.map1x, self.frontend.map1y, cv2.INTER_LINEAR)
                    gt_image = torch.from_numpy(gt_image).to("cuda").float() / 255.0  # 转换为 PyTorch 张量并归一化
                    gt_image = gt_image.permute(2, 0, 1)  # transfer to K opt

                    pose_optimizers = torch.optim.Adam(pose_opt_params)
                    for iter in range(200):
                        pose_optimizers.zero_grad()
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background,kernel_size=self.kernel_size
                        )
                        image, visibility_filter, radii = (
                            render_pkg["render"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                        )

                        #gt_image = viewpoint.original_image.cuda()
                        Ll1 = l1_loss(image, gt_image)
                        loss = (1.0 - self.opt_params.lambda_dssim) * (
                            Ll1
                        ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        loss.backward()


                        with torch.no_grad():
                            pose_optimizers.step()
                            update_pose(viewpoint)
                        # if converged:
                        #     break

                        viewpoint.original_image = None
                        if i%10 == 0:
                            torch.cuda.empty_cache()

                



                ATE = eval_ate(
                    self.frontend.cameras,
                    self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )

                rendering_result = eval_rendering_vings_mip(
                    self.frontend.cameras,
                    self.input_folder,
                    self.gaussians,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    map1x = self.map1x,
                    map1y = self.map1y,
                    iteration="after_opt_pose",
                    kernel_size = self.kernel_size
                )
                metrics_table.add_data(
                    "After",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )
                wandb.log({"Metrics": metrics_table})
                #save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

            self.backend_queue.put(["stop"])
            backend_process.join()
            Log("Backend stopped and joined the main thread")
            if self.use_gui:
                self.q_main2vis.put(gui_utils.GaussianPacket_mip(finish=True))
                gui_process.join()
                Log("GUI Stopped and joined the main thread")

        except  KeyboardInterrupt:
            self.shutdown_event.set()
            self.backend_queue.put(["stop"])
            backend_process.terminate()
            if self.use_gui:
                gui_process.terminate()


if __name__ == '__main__':
    
    # args = parser.parse_args()

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        print(path)
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_vinsgsstereo_" + path[-2] + "_" + path[-1], current_datetime
        )
        print(save_dir)
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="Vinsgsstereo",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")


    
    rospy.init_node('gsestimator', anonymous=True)

    rospy.Subscriber("/pose_graph/pose_graph_path", Path, pose_graph_callback , queue_size=20)
    rospy.Subscriber("/pose_graph/pose_update", Header, pose_update_callback , queue_size=20)
    rospy.Subscriber("/vins_estimator/odometry", Odometry, odometry_callback , queue_size=20)
    rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback , queue_size=20)


    shutdown_event = Event()
    slam_vinsgs = vinsgs(config , save_dir=save_dir)


    process_thread = threading.Thread(target=slam_vinsgs.process)
    process_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        Log("收到KeyboardInterrupt信号，正在关闭。")
    finally:
        shutdown_event.set()
        process_thread.join(timeout=5)
        if process_thread.is_alive():
            Log("处理线程未能及时终止，强制退出。")
            sys.exit(1)
        Log("关闭完成。")
