#!/usr/bin/env python3
"""
Animated PLY Viewer
Plays back a sequence of PLY files from a directory.

Usage:
    python ply_viewer.py --folder sim_results/sand --fps 30
"""

import argparse
import glob
import os
import time
from pathlib import Path

try:
    import open3d as o3d
    import numpy as np
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def read_ply_simple(filename):
    """
    Simple PLY reader that returns points as numpy array.
    Works without open3d dependency.
    """
    points = []
    colors = None

    with open(filename, 'rb') as f:
        # Read header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")

        vertex_count = 0
        has_color = False
        is_binary = False

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property') and ('red' in line or 'r' == line.split()[-1]):
                has_color = True
            elif line == 'end_header':
                break

        # Read vertices
        if is_binary:
            # Simple binary reader (assumes float32 for x,y,z and optionally uint8 for r,g,b)
            import struct
            if has_color:
                fmt = 'fffBBB'
                size = struct.calcsize(fmt)
                colors = []
                for _ in range(vertex_count):
                    data = struct.unpack(fmt, f.read(size))
                    points.append(data[:3])
                    colors.append([data[3]/255.0, data[4]/255.0, data[5]/255.0])
                colors = np.array(colors)
            else:
                fmt = 'fff'
                size = struct.calcsize(fmt)
                for _ in range(vertex_count):
                    points.append(struct.unpack(fmt, f.read(size)))
        else:
            # ASCII format
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip().split()
                points.append([float(line[0]), float(line[1]), float(line[2])])
                if has_color and len(line) >= 6:
                    if colors is None:
                        colors = []
                    colors.append([float(line[3])/255.0, float(line[4])/255.0, float(line[5])/255.0])

            if colors:
                colors = np.array(colors)

    return np.array(points), colors


class PLYViewer:
    def __init__(self, folder_path, fps=30, loop=True):
        """
        Initialize the PLY viewer.

        Args:
            folder_path: Path to folder containing PLY files
            fps: Frames per second for playback
            loop: Whether to loop the animation
        """
        self.folder_path = Path(folder_path)
        self.fps = fps
        self.loop = loop
        self.frame_delay = 1.0 / fps

        # Find all PLY files and sort them
        self.ply_files = sorted(glob.glob(str(self.folder_path / "*.ply")))

        if not self.ply_files:
            raise ValueError(f"No PLY files found in {folder_path}")

        print(f"Found {len(self.ply_files)} PLY files")
        print(f"Playing at {fps} FPS")
        print(f"Looping: {loop}")

    def play_open3d(self):
        """Play animation using Open3D visualizer."""
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D is required for visualization")

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PLY Animation Viewer")

        # Load first frame to initialize
        first_pcd = o3d.io.read_point_cloud(self.ply_files[0])
        vis.add_geometry(first_pcd)

        # Get render options
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        # Set up camera
        view_control = vis.get_view_control()

        print("\nControls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Space: Pause/Resume")
        print("  - R: Reset camera")
        print("  - Q/Esc: Quit")
        print("\nPlaying animation...")

        frame_idx = 0
        paused = False
        last_time = time.time()

        # Animation loop
        while True:
            current_time = time.time()
            elapsed = current_time - last_time

            # Update frame if enough time has passed and not paused
            if not paused and elapsed >= self.frame_delay:
                # Load and update point cloud
                pcd = o3d.io.read_point_cloud(self.ply_files[frame_idx])
                first_pcd.points = pcd.points
                if pcd.has_colors():
                    first_pcd.colors = pcd.colors
                if pcd.has_normals():
                    first_pcd.normals = pcd.normals

                vis.update_geometry(first_pcd)

                # Print frame info
                frame_name = os.path.basename(self.ply_files[frame_idx])
                print(f"\rFrame {frame_idx + 1}/{len(self.ply_files)}: {frame_name}", end="", flush=True)

                # Move to next frame
                frame_idx += 1
                if frame_idx >= len(self.ply_files):
                    if self.loop:
                        frame_idx = 0
                        print("\n[Looping animation...]")
                    else:
                        print("\n[Animation complete]")
                        break

                last_time = current_time

            # Update visualizer
            if not vis.poll_events():
                break
            vis.update_renderer()

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

        vis.destroy_window()
        print("\nViewer closed.")

    def play_matplotlib(self):
        """Play animation using matplotlib (fallback when Open3D not available)."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Either Open3D or Matplotlib is required for visualization")

        print("\nUsing Matplotlib viewer (simpler, slower)")
        print("Note: For better performance, install Open3D with: pip install open3d")
        print("\nControls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Close window to exit")
        print("\nPlaying animation...")

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Load first frame to set up plot
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(self.ply_files[0])
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        else:
            points, colors = read_ply_simple(self.ply_files[0])

        # Set up plot limits
        bounds = np.max(np.abs(points), axis=0)
        ax.set_xlim(-bounds[0], bounds[0])
        ax.set_ylim(-bounds[1], bounds[1])
        ax.set_zlim(-bounds[2], bounds[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Initial scatter plot
        if colors is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=colors, s=1, marker='.')
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c='blue', s=1, marker='.')

        frame_idx = [0]  # Use list to allow modification in nested function

        def update_frame():
            """Update to next frame."""
            if frame_idx[0] >= len(self.ply_files):
                if self.loop:
                    frame_idx[0] = 0
                    print("\n[Looping animation...]")
                else:
                    print("\n[Animation complete]")
                    plt.close()
                    return

            # Load new frame
            if HAS_OPEN3D:
                pcd = o3d.io.read_point_cloud(self.ply_files[frame_idx[0]])
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            else:
                points, colors = read_ply_simple(self.ply_files[frame_idx[0]])

            # Update scatter plot
            scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            if colors is not None:
                scatter.set_color(colors)

            # Update title
            frame_name = os.path.basename(self.ply_files[frame_idx[0]])
            ax.set_title(f"Frame {frame_idx[0] + 1}/{len(self.ply_files)}: {frame_name}")

            print(f"\rFrame {frame_idx[0] + 1}/{len(self.ply_files)}: {frame_name}", end="", flush=True)

            frame_idx[0] += 1
            fig.canvas.draw_idle()

        # Set up animation timer
        timer = fig.canvas.new_timer(interval=int(self.frame_delay * 1000))
        timer.add_callback(update_frame)
        timer.start()

        plt.show()
        print("\nViewer closed.")

    def play(self):
        """Play animation using the best available backend."""
        if HAS_OPEN3D:
            print("Using Open3D viewer (recommended)")
            self.play_open3d()
        elif HAS_MATPLOTLIB:
            print("Open3D not available, using Matplotlib viewer")
            self.play_matplotlib()
        else:
            raise RuntimeError("No visualization backend available. Please install open3d or matplotlib.")

    def export_video(self, output_path="animation.mp4", resolution=(1920, 1080)):
        """
        Export animation to video file.

        Args:
            output_path: Path for output video file
            resolution: Video resolution (width, height)
        """
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D is required for video export")

        print(f"\nExporting video to {output_path}...")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"FPS: {self.fps}")

        # Create visualizer for offscreen rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1], visible=False)

        # Load first frame
        pcd = o3d.io.read_point_cloud(self.ply_files[0])
        vis.add_geometry(pcd)

        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        # Render all frames and save as images
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        for idx, ply_file in enumerate(self.ply_files):
            pcd_new = o3d.io.read_point_cloud(ply_file)
            pcd.points = pcd_new.points
            if pcd_new.has_colors():
                pcd.colors = pcd_new.colors
            if pcd_new.has_normals():
                pcd.normals = pcd_new.normals

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Save frame
            frame_path = temp_dir / f"frame_{idx:06d}.png"
            vis.capture_screen_image(str(frame_path))
            print(f"\rRendering frame {idx + 1}/{len(self.ply_files)}", end="", flush=True)

        vis.destroy_window()
        print("\n")

        # Use ffmpeg to create video
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video exported successfully to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
            print("Frames saved in temp_frames/ directory")
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to create video.")
            print("Frames saved in temp_frames/ directory")

        # Clean up temp frames
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Animated PLY file viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View sand simulation at 30 FPS
  python ply_viewer.py --folder sim_results/sand --fps 30
  
  # View without looping
  python ply_viewer.py --folder sim_results/sand --no-loop
  
  # Export to video
  python ply_viewer.py --folder sim_results/sand --export animation.mp4
        """
    )

    parser.add_argument(
        "--folder", "-f",
        type=str,
        default="sim_results/sand",
        help="Folder containing PLY files (default: sim_results/sand)"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Frames per second for playback (default: 30)"
    )

    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Don't loop the animation"
    )

    parser.add_argument(
        "--export", "-e",
        type=str,
        metavar="OUTPUT",
        help="Export animation to video file instead of displaying"
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Video resolution for export (default: 1920x1080)"
    )

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using 1920x1080")
        resolution = (1920, 1080)

    # Create viewer
    try:
        viewer = PLYViewer(
            folder_path=args.folder,
            fps=args.fps,
            loop=not args.no_loop
        )

        if args.export:
            viewer.export_video(output_path=args.export, resolution=resolution)
        else:
            viewer.play_open3d()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

