import open3d as o3d 
import numpy as np

rgb_raw = np.ones((480, 640, 3))*0.5*255
depthmap = np.zeros((480, 640))
depthmap[:10, :10] = 1
rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
depth = o3d.geometry.Image(depthmap.astype(np.float32))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

fx = 450 ; fy = 450 ; cx = 320 ; cy = 240

W2C = np.eye(4)
#time1 = time.time()
pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd,
    o3d.camera.PinholeCameraIntrinsic(
        480,
        640,
        fx,
        fy,
        cx,
        cy,
    ),
    extrinsic=W2C,
    project_valid_depth_only=True,
)
print(pcd_tmp)