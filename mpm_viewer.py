import queue
import threading
import time

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


class MPM_Viewer:
    def __init__(self):
        self.particle_pos = None
        self.particle_vel = None
        self.particle_stress = None
        self.target_fps = 60
        self.message_queue = queue.Queue() # a thread-safe queue to receive data from the main thread

    def launch_window(self):
        # check that we're on the main thread
        assert(threading.current_thread() == threading.main_thread())
        # launch a window and start the rendering loop, which calls self.update() every frame
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        pcd_data = o3d.data.DemoICPPointClouds()
        self.pcd = o3d.io.read_point_cloud(pcd_data.paths[0])
        #self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        last_frame_time = 0
        while True:
            # throttle frame rate
            while time.time() - last_frame_time < 1.0 / self.target_fps:
                time.sleep(0.001)
            last_frame_time = time.time()

            self.update()
            self.vis.poll_events()
            self.vis.update_renderer()


    def update_data(self, position: np.array, velocity: np.array, stress: np.array):
        # update the particle data to be visualized, called from the main thread
        self.message_queue.put(dict(
            type='update_particles',
            position=position,
            velocity=velocity,
            stress=stress)
        )

    def update(self):
        # draw the particles as points, with color based on velocity or stress
        if not self.message_queue.empty():
            message = self.message_queue.get()
            if message['type'] == 'update_particles':
                self.particle_pos = message['position']
                self.particle_vel = message['velocity']
                self.particle_stress = message['stress']
                self._update_scene()

    def _update_scene(self):
        # update vis scene from particle_pos, particle_vel, particle_stress
        self.pcd.points = o3d.utility.Vector3dVector(self.particle_pos)
        # color by velocity magnitude
        vel_mag = (self.particle_vel ** 2).sum(axis=1) ** 0.5
        vel_mag_normalized = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-8)
        colors = o3d.utility.Vector3dVector(plt.cm.viridis(vel_mag_normalized)[:,:3]) # use viridis colormap
        self.pcd.colors = colors
        self.vis.update_geometry(self.pcd)

    #def update_camera(self):
    #    # use a game-like input paradigm (wasd/arrow keys to move, mouse to look around) to update the camera position and orientation

