#!/home/wkx123/anaconda3/envs/splatam/bin/python
# -*- coding: utf-8 -*-
import rospy
import torch
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import PointCloud, Image
from visualization_msgs.msg import Marker, MarkerArray
from queue import Queue, Empty
import sys
from threading import Lock, Condition, Thread, Event
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
from utils.vinsgs_backend_BA import BackEnd
from utils.vinsgs_frontend_stereo_BA import Vins_FrontEnd
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering_vings, save_gaussians
from utils.logging_utils import TicToc
import time
import wandb
from utils.pose_utils import update_pose
import glob
import torch
import cv2
import torch.nn as nn

last_imu_t = 0.0

# 初始化队列
imu_odom_buf = Queue()
point_cloud_buf = Queue()
margin_point_cloud_buf = Queue()
campose_buf = Queue()
keypose_buf = Queue()   # Camera keyposes   
odometry_buf = Queue()
key_point_cloud_buf = Queue()
key_point_cloud_incre_buf = Queue()
pose_graph_buf = Queue()
pose_update_buf = Queue()

# 初始化锁和条件变量
m_buf = Lock()
con = Condition(m_buf)

def imu_odom_callback(data):
    imu_odom_buf.put(data)
    with con:
        con.notify()

def point_cloud_callback(data):
    point_cloud_buf.put(data)
    with con:
        con.notify()

def margin_cloud_callback(data):
    margin_point_cloud_buf.put(data)
    with con:
        con.notify()

def campose_callback(data):
    campose_buf.put(data)
    with con:
        con.notify()

def odometry_callback(data):
    odometry_buf.put(data)
    with con:
        con.notify()

def keypose_callback(data):
    keypose_buf.put(data)
    with con:
        con.notify()

def key_point_cloud_callback(data):
    key_point_cloud_buf.put(data)
    with con:
        con.notify()

def pose_graph_callback(data):
    pose_graph_buf.put(data)
    with con:
        con.notify()

def pose_update_callback(data):
    pose_update_buf.put(data)
    with con:
        con.notify()

class VinsGS:
    def __init__(self, config, save_dir=None, shutdown_event=None):
        self.config = config
        self.save_dir = save_dir
        self.timeout = 5.0
        self.shutdown_event = shutdown_event or Event()

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
        self.map1x, self.map1y = self.frontend.map1x, self.frontend.map1y

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

    def process(self):
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            backend_process = mp.Process(target=self.backend.run)
            backend_process.daemon = True  # 确保 backend_process 随主程序退出
            if self.use_gui:
                gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
                gui_process.daemon = True  # 确保 gui_process 随主程序退出
                gui_process.start()
                time.sleep(1)

            backend_process.start()
            started = False
            last_message_time = time.time()

            while not rospy.is_shutdown() and not self.shutdown_event.is_set():
                with con:
                    # 等待通知或超时
                    notified = con.wait(timeout=0.1)

                measurements = self.frontend.get_measurements2(
                    pose_graph_buf, pose_update_buf, odometry_buf, keypose_buf
                )
                if measurements:
                    self.frontend.run(measurements)
                    last_message_time = time.time()
                    started = True
                else:
                    if not started:
                        continue
                    if time.time() - last_message_time > self.timeout:
                        print("超时，退出循环。")
                        break

            # 开始关闭进程
            self.backend_queue.put(["pause"])
            end.record()
            torch.cuda.synchronize()

            N_frames = len(self.frontend.cameras)
            FPS = N_frames / (start.elapsed_time(end) * 0.001)
            Log("总时间", start.elapsed_time(end) * 0.001, tag="Eval")
            Log("总FPS", FPS, tag="Eval")

            if self.eval_rendering:
                print("评估渲染...")
                self.evaluate_rendering(FPS)

            # 向后端进程发送停止信号
            self.backend_queue.put(["stop"])
            backend_process.join(timeout=5)
            if backend_process.is_alive():
                backend_process.terminate()
                Log("后端进程被强制终止。")

            Log("后端已停止并加入主线程")

            if self.use_gui:
                self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                gui_process.join(timeout=5)
                if gui_process.is_alive():
                    gui_process.terminate()
                    Log("GUI进程被强制终止。")
                Log("GUI已停止并加入主线程")

        except Exception as e:
            Log(f"process线程中出现异常: {e}")
            self.backend_queue.put(["stop"])
            backend_process.terminate()
            if self.use_gui:
                gui_process.terminate()
            raise e

    def evaluate_rendering(self, FPS):
        # 此处插入你现有的渲染评估代码
        pass

def shutdown_callback(shutdown_event, slam_vinsgs):
    Log("收到关闭信号。")
    shutdown_event.set()
    # 选做：在此执行额外的清理操作
    slam_vinsgs.backend_queue.put(["stop"])
    if slam_vinsgs.use_gui:
        slam_vinsgs.q_main2vis.put(gui_utils.GaussianPacket(finish=True))

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True, help="配置YAML文件的路径。")
    parser.add_argument("--eval", action="store_true", help="以评估模式运行。")
    
    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("以评估模式运行 MonoGS")
        Log("以下配置将被覆盖：")
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
        path = config["Dataset"]["dataset_path"].strip("/").split("/")
        print(path)
        save_dir = os.path.join(
            config["Results"]["save_dir"], 
            f"{path[-3]}_vinsgsstereo_{path[-2]}_{path[-1]}", 
            current_datetime
        )
        print(save_dir)
        tmp = os.path.splitext(os.path.basename(args.config))[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log(f"保存结果到 {save_dir}")
        run = wandb.init(
            project="Vinsgsstereo",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    rospy.init_node('gsestimator', anonymous=True)

    rospy.Subscriber("/pose_graph/pose_graph_path", Path, pose_graph_callback, queue_size=20)
    rospy.Subscriber("/pose_graph/pose_update", Header, pose_update_callback, queue_size=20)
    rospy.Subscriber("/vins_estimator/odometry", Odometry, odometry_callback, queue_size=20)
    rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback, queue_size=20)

    shutdown_event = Event()
    slam_vinsgs = VinsGS(config, save_dir=save_dir, shutdown_event=shutdown_event)

    # 将处理线程作为守护线程启动
    process_thread = Thread(target=slam_vinsgs.process, daemon=True)
    process_thread.start()

    # 注册关闭回调
    rospy.on_shutdown(lambda: shutdown_callback(shutdown_event, slam_vinsgs))

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
