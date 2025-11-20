import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import os
import io
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import imageio
except ImportError:
    imageio = None

#################################################################################
#                                   Data Params                                 #
#################################################################################
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

kit_raw_offsets = np.array(
    [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
     [1, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0],
     [0, -1, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1]])
t2m_raw_offsets = np.array([[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0],
                            [0,-1,0], [0,1,0], [0,-1,0], [0,-1,0], [0,1,0],
                            [0,0,1], [0,0,1], [0,1,0], [1,0,0], [-1,0,0],
                            [0,0,1], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0],
                            [0,-1,0], [0,-1,0]])

#################################################################################
#                                  Joints Revert                                #
#################################################################################
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

#################################################################################
#                                 Motion Plotting                               #
#################################################################################
def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4, save_frames_dir=None):
    # Ensure Agg backend is used (already set at module level, but ensure it's active)
    if matplotlib.get_backend() != 'Agg':
        matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # Ensure joints is in the correct format: (seq_len, num_joints, 3)
    joints = np.array(joints)
    if joints.ndim == 3:
        # Already in (seq_len, num_joints, 3) format
        data = joints.copy()
    elif joints.ndim == 2:
        # If 2D, reshape to (seq_len, num_joints, 3)
        # Assume it's (seq_len * num_joints, 3) or (seq_len, num_joints * 3)
        if joints.shape[1] == 3:
            # (seq_len * num_joints, 3) - need to infer num_joints
            # For t2m, we expect 22 joints
            num_joints = 22
            seq_len = joints.shape[0] // num_joints
            data = joints.reshape(seq_len, num_joints, 3)
        else:
            # (seq_len, num_joints * 3)
            num_joints = joints.shape[1] // 3
            data = joints.reshape(len(joints), num_joints, 3)
    else:
        raise ValueError(f"Invalid joints shape: {joints.shape}, expected (seq_len, num_joints, 3)")
    
    # Check if data is valid
    if data.size == 0:
        raise ValueError("Invalid motion data: data is empty")
    
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    # Recompute bounds after centering
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    # Add some padding
    center = (MINS + MAXS) / 2
    ranges = MAXS - MINS
    # Ensure we have a minimum range to avoid issues with very small or zero ranges
    min_range = 0.1  # Minimum range for each axis
    ranges = np.maximum(ranges, min_range)
    max_range = max(ranges) * 1.2  # 20% padding
    plot_mins = center - max_range / 2
    plot_maxs = center + max_range / 2
    

    def update(index):
        # Clear axes properly
        ax.cla()
        # Reapply title
        fig.suptitle(title, fontsize=20)
        # Reapply view settings and limits based on actual data bounds
        ax.set_xlim3d([plot_mins[0], plot_maxs[0]])
        ax.set_ylim3d([plot_mins[1], plot_maxs[1]])
        ax.set_zlim3d([plot_mins[2], plot_maxs[2]])
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        ax.grid(False)
        plot_xzPlane(plot_mins[0] - trajec[index, 0], plot_maxs[0] - trajec[index, 0], 0, 
                     plot_mins[2] - trajec[index, 1], plot_maxs[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < len(colors):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                # Ensure chain indices are valid
                valid_chain = [idx for idx in chain if idx < data.shape[1]]
                if len(valid_chain) > 1:
                    ax.plot3D(data[index, valid_chain, 0], data[index, valid_chain, 1], data[index, valid_chain, 2], 
                             linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    def render_frame(frame_idx):
        """Render a single frame and return as PIL Image"""
        # Create a fresh figure for each frame to avoid 3D projection issues
        frame_fig = plt.figure(figsize=figsize, facecolor='white')
        frame_ax = frame_fig.add_subplot(111, projection='3d')
        frame_fig.suptitle(title, fontsize=20)
        
        # Set limits and view
        frame_ax.set_xlim3d([plot_mins[0], plot_maxs[0]])
        frame_ax.set_ylim3d([plot_mins[1], plot_maxs[1]])
        frame_ax.set_zlim3d([plot_mins[2], plot_maxs[2]])
        frame_ax.view_init(elev=120, azim=-90)
        frame_ax.dist = 7.5
        frame_ax.grid(False)
        
        # Plot ground plane
        minx = plot_mins[0] - trajec[frame_idx, 0]
        maxx = plot_maxs[0] - trajec[frame_idx, 0]
        minz = plot_mins[2] - trajec[frame_idx, 1]
        maxz = plot_maxs[2] - trajec[frame_idx, 1]
        verts = [[minx, 0, minz], [minx, 0, maxz], [maxx, 0, maxz], [maxx, 0, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        frame_ax.add_collection3d(xz_plane)
        
        # Plot trajectory
        if frame_idx > 1:
            frame_ax.plot3D(trajec[:frame_idx, 0] - trajec[frame_idx, 0], np.zeros_like(trajec[:frame_idx, 0]),
                      trajec[:frame_idx, 1] - trajec[frame_idx, 1], linewidth=1.0,
                      color='blue')
        
        # Plot skeleton
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < len(colors):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                valid_chain = [idx for idx in chain if idx < data.shape[1]]
                if len(valid_chain) > 1:
                    frame_ax.plot3D(data[frame_idx, valid_chain, 0], data[frame_idx, valid_chain, 1], 
                                   data[frame_idx, valid_chain, 2], linewidth=linewidth, color=color)
        
        plt.axis('off')
        frame_ax.set_xticklabels([])
        frame_ax.set_yticklabels([])
        frame_ax.set_zticklabels([])
        
        # Convert to image - copy data so it's independent of the buffer
        buf = io.BytesIO()
        frame_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        buf.seek(0)
        # Create image and convert to RGB for GIF compatibility
        img = Image.open(buf)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_copy = img.copy()  # Create a copy that's independent of the buffer
        buf.close()
        plt.close(frame_fig)
        return img_copy

    # Always use frame-by-frame rendering for reliability (works with both GIF and MP4)
    # This ensures 3D plots render correctly
    actual_path = save_path
    
    # Note: We'll use frame-by-frame rendering for both MP4 and GIF
    # imageio-ffmpeg will be used for MP4 if available
    
    # Use frame-by-frame rendering (works reliably for GIF)
    if Image is None:
        raise RuntimeError("PIL/Pillow is required for GIF generation. Please install: pip install Pillow")
    
    # Use provided frames directory or create one for debugging
    frames_dir = save_frames_dir
    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to: {frames_dir}")
    
    frames = []
    print(f"Rendering {frame_number} frames...")
    for i in range(frame_number):
        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}/{frame_number}")
        frame_img = render_frame(i)
        frames.append(frame_img)
        
        # Save individual frame as PNG for debugging
        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            if i == 0 or i == frame_number - 1:
                print(f"    Saved frame {i} to {frame_path}")
    
    # Save video - prefer MP4 if imageio-ffmpeg is available, otherwise GIF
    if len(frames) > 0:
        # Ensure all frames are in the same mode and size
        frames_rgb = []
        first_frame = frames[0]
        if first_frame.mode != 'RGB':
            first_frame = first_frame.convert('RGB')
        
        # Get size from first frame
        target_size = first_frame.size
        frames_rgb.append(first_frame)
        
        # Convert and resize all other frames to match
        for frame in frames[1:]:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            # Ensure all frames are the same size
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)
            frames_rgb.append(frame)
        
        # Convert PIL Images to numpy arrays for imageio
        frame_arrays = [np.array(frame) for frame in frames_rgb]
        
        # Try to save as MP4 first (better quality and compatibility)
        if actual_path.endswith('.mp4'):
            if imageio is not None:
                try:
                    # Use imageio-ffmpeg for MP4 (automatically uses ffmpeg if available)
                    imageio.mimsave(actual_path, frame_arrays, fps=fps, codec='libx264', quality=8)
                    print(f"Saved {len(frames_rgb)} frames to MP4 using imageio-ffmpeg: {actual_path}")
                except Exception as e:
                    print(f"Error saving MP4 with imageio-ffmpeg: {e}")
                    print("Falling back to GIF format...")
                    # Fall back to GIF
                    base_path = os.path.splitext(actual_path)[0]
                    actual_path = base_path + '.gif'
                    if imageio is not None:
                        try:
                            imageio.mimsave(actual_path, frame_arrays, duration=1.0/fps, loop=0)
                            print(f"Saved {len(frames_rgb)} frames to GIF using imageio: {actual_path}")
                        except Exception as e2:
                            print(f"imageio GIF failed, using PIL: {e2}")
                            frames_rgb[0].save(
                                actual_path,
                                save_all=True,
                                append_images=frames_rgb[1:],
                                duration=int(1000 / fps),
                                loop=0,
                                optimize=False
                            )
                    else:
                        frames_rgb[0].save(
                            actual_path,
                            save_all=True,
                            append_images=frames_rgb[1:],
                            duration=int(1000 / fps),
                            loop=0,
                            optimize=False
                        )
            else:
                print("imageio not available, cannot save MP4. Falling back to GIF...")
                base_path = os.path.splitext(actual_path)[0]
                actual_path = base_path + '.gif'
                frames_rgb[0].save(
                    actual_path,
                    save_all=True,
                    append_images=frames_rgb[1:],
                    duration=int(1000 / fps),
                    loop=0,
                    optimize=False
                )
        elif actual_path.endswith('.gif'):
            # Save as GIF
            if imageio is not None:
                try:
                    imageio.mimsave(actual_path, frame_arrays, duration=1.0/fps, loop=0)
                    print(f"Saved {len(frames_rgb)} frames to GIF using imageio: {actual_path}")
                except Exception as e:
                    print(f"imageio failed, using PIL: {e}")
                    frames_rgb[0].save(
                        actual_path,
                        save_all=True,
                        append_images=frames_rgb[1:],
                        duration=int(1000 / fps),
                        loop=0,
                        optimize=False
                    )
            else:
                frames_rgb[0].save(
                    actual_path,
                    save_all=True,
                    append_images=frames_rgb[1:],
                    duration=int(1000 / fps),
                    loop=0,
                    optimize=False
                )
    
    if frames_dir is not None:
        print(f"Saved {len(frames)} individual frames to {frames_dir}")
    
    return actual_path
