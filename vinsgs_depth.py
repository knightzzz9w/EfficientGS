#!/home/wkx123/anaconda3/envs/splatam/bin/python
# -*- coding: utf-8 -*-
import rospy
import torch
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud , Image
from visualization_msgs.msg import Marker, MarkerArray
from queue import Queue
import sys
from threading import Lock, Condition
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
import threading
from argparse import ArgumentParser
from datetime import datetime
import yaml
import torch.multiprocessing as mp
import os
from munch import munchify
from utils.multiprocessing_utils import FakeQueue
from utils.vinsgs_backend import BackEnd
from utils.vinsgs_frontend_depth import Vins_FrontEnd
from utils.logging_utils import Log
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering_vings, save_gaussians
from utils.logging_utils import TicToc
import time
import wandb

last_imu_t = 0.0

image_buf = Queue()
imu_odom_buf = Queue()
point_cloud_buf = Queue()
margin_point_cloud_buf = Queue()
campose_buf = Queue()
keypose_buf = Queue()   #camera的keyposes   
key_point_cloud_buf = Queue()
key_point_cloud_incre_buf = Queue()



m_buf = Lock()
con = Condition(m_buf)

def image_callback(data):
    with con:
        image_buf.put(data)
        con.notify()  

def imu_odom_callback(data):
    with con:
        imu_odom_buf.put(data)
        con.notify()  

def point_cloud_callback(data):
    with con:
        point_cloud_buf.put(data)
        con.notify()  

def margin_cloud_callback(data):
    with con:
        margin_point_cloud_buf.put(data)
        con.notify()  

def campose_callback(data):
    with con:
        campose_buf.put(data)
        con.notify()  

def keypose_callback(data):
    with con:
        keypose_buf.put(data)
        con.notify()
        
def key_point_cloud_callback(data):
    with con:
        key_point_cloud_buf.put(data)
        con.notify()

def key_point_cloud_incre_callback(data):
    with con:
        key_point_cloud_incre_buf.put(data)
        con.notify()


class vinsgs:
    def __init__(self, config, save_dir=None):
        
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

        self.frontend_queue = mp.Queue()
        self.backend_queue = mp.Queue()

        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = Vins_FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        self.map1x, self.map1y = self.frontend.map1x , self.frontend.map1y

        
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = self.frontend_queue
        self.frontend.backend_queue = self.backend_queue
        self.frontend.q_main2vis = self.q_main2vis
        self.frontend.q_vis2main = self.q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = self.frontend_queue
        self.backend.backend_queue = self.backend_queue
        self.backend.live_mode = self.live_mode


        self.frontend_queue = self.frontend_queue
        self.backend_queue = self.backend_queue

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )



    def process(self) :

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(1)

        backend_process.start()   ; started = False
        while not rospy.is_shutdown():
            with con:
                measurements = self.frontend.get_measurements3(image_buf , campose_buf , point_cloud_buf)
                
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
            print("cam dict is " , self.frontend.cameras.keys())
            rendering_result = eval_rendering_vings(
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

            rendering_result = eval_rendering_vings(
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
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        self.backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")
            

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
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_vinsgsmonodepth_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="Vinsgsmono",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")


    
    rospy.init_node('gsestimator', anonymous=True)

    rospy.Subscriber("/cam0/image_raw", Image, image_callback , queue_size=100)    #real time rendering
    rospy.Subscriber("/vins_estimator/camera_pose", Odometry, campose_callback , queue_size=50)
    #rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback , queue_size=10)
    #rospy.Subscriber("/vins_estimator/keyframe_point", PointCloud, key_point_cloud_callback , queue_size=50)
    #rospy.Subscriber("/vins_estimator/keyframe_point_incre", PointCloud, key_point_cloud_incre_callback , queue_size=50)
    
    rospy.Subscriber("/vins_estimator/point_cloud", PointCloud, point_cloud_callback , queue_size=50)
    # rospy.Subscriber("/cam0/image_raw", Image, image_callback )   #no need for real time rendering
    # rospy.Subscriber("/vins_estimator/camera_pose", Odometry, campose_callback )
    # rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback )
   
    # #rospy.Subscriber("/vins_estimator/keyframe_point_incre", PointCloud, key_point_cloud_incre_callback )

    slam_vinsgs = vinsgs(config , save_dir=save_dir)


    process_thread = threading.Thread(target=slam_vinsgs.process)
    process_thread.start()

    rospy.spin()
    process_thread.join()