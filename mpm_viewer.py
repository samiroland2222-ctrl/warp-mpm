import queue
import threading
import time

import numpy as np
import viser
from matplotlib import pyplot as plt


class MPM_Viewer:
    def __init__(self):
        self.particle_pos = None
        self.particle_vel = None
        self.particle_stress = None
        self.target_fps = 60
        self.message_queue = queue.Queue() # a thread-safe queue to receive data from the main thread
        self.server = None
        self.point_cloud_handle = None

    def launch_window(self, blocking=True):
        # check that we're on the main thread
        assert(threading.current_thread() == threading.main_thread())
        # launch viser server and start the rendering loop
        self.server = viser.ViserServer()
        print(f"Viser server started at: http://localhost:{self.server.get_port()}")

        # Initialize with empty point cloud
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            name="/particles",
            points=np.zeros((1, 3)),
            colors=np.array([[0, 0, 255]]),
            point_size=0.01,
        )

        if not blocking:
            return

        last_frame_time = 0
        try:
            while True:
                # throttle frame rate
                while time.time() - last_frame_time < 1.0 / self.target_fps:
                    time.sleep(0.001)
                last_frame_time = time.time()

                self.update()
        except KeyboardInterrupt:
            print("Viewer closed")


    def update_data(self, position: np.ndarray, velocity: np.ndarray, stress: np.ndarray):
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
        if self.particle_pos is None or len(self.particle_pos) == 0:
            return

        # color by velocity magnitude
        vel_mag = (self.particle_vel ** 2).sum(axis=1) ** 0.5
        vel_mag_normalized = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-8)
        colors = plt.cm.viridis(vel_mag_normalized)[:,:3] # use viridis colormap

        # Convert to uint8 RGB colors (0-255)
        colors_rgb = (colors * 255).astype(np.uint8)

        # Update the point cloud
        if self.point_cloud_handle is not None:
            self.point_cloud_handle.points = self.particle_pos
            self.point_cloud_handle.colors = colors_rgb

    #def update_camera(self):
    #    # use a game-like input paradigm (wasd/arrow keys to move, mouse to look around) to update the camera position and orientation

