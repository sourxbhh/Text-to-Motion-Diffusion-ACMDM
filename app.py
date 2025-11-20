"""
Gradio Web Interface for ACMDM Motion Generation
Milestone 3: User-friendly web interface for text-to-motion generation
"""

import os
import sys
from os.path import join as pjoin, dirname, abspath

# Add current directory to Python path for HuggingFace Spaces
# This ensures models/ and utils/ can be imported
current_dir = dirname(abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import numpy as np
import gradio as gr
from typing import Optional, Tuple, List
import tempfile
import random

# Import from models and utils
from models.AE_2D_Causal import AE_models
from models.ACMDM import ACMDM_models
from models.LengthEstimator import LengthEstimator
from utils.back_process import back_process
from utils.motion_process import plot_3d_motion, t2m_kinematic_chain

# Global variables for model caching
_models_cache = {
    'ae': None,
    'acmdm': None,
    'length_estimator': None,
    'stats': None,
    'device': None,
    'loaded': False
}


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def load_models_cached(
    gpu_id: int = 0,
    model_name: str = 'ACMDM_Flow_S_PatchSize22',
    ae_name: str = 'AE_2D_Causal',
    ae_model: str = 'AE_Model',
    model_type: str = 'ACMDM-Flow-S-PatchSize22',
    dataset_name: str = 't2m',
    checkpoints_dir: str = './checkpoints',
    use_length_estimator: bool = False
) -> Tuple[dict, str]:
    """
    Load models with caching to avoid reloading on every request.
    Returns (models_dict, status_message)
    """
    global _models_cache
    
    # Check if models are already loaded
    if _models_cache['loaded']:
        return _models_cache, "Models already loaded (using cache)"
    
    try:
        # Determine device
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        
        _models_cache['device'] = device
        status_messages = [f"Using device: {device}"]
        
        # Load AE
        status_messages.append("Loading AE model...")
        ae = AE_models[ae_model](input_width=3)
        ae_ckpt_path = pjoin(checkpoints_dir, dataset_name, ae_name, 'model', 'latest.tar')
        
        if not os.path.exists(ae_ckpt_path):
            return None, f"Error: AE checkpoint not found at {ae_ckpt_path}"
        
        ae_ckpt = torch.load(ae_ckpt_path, map_location='cpu')
        ae.load_state_dict(ae_ckpt['ae'])
        ae.eval()
        ae.to(device)
        _models_cache['ae'] = ae
        status_messages.append(f"✓ Loaded AE from {ae_ckpt_path}")
        
        # Load ACMDM
        status_messages.append("Loading ACMDM model...")
        acmdm = ACMDM_models[model_type](input_dim=ae.output_emb_width, cond_mode='text')
        acmdm_ckpt_path = pjoin(checkpoints_dir, dataset_name, model_name, 'model', 'latest.tar')
        
        if not os.path.exists(acmdm_ckpt_path):
            return None, f"Error: ACMDM checkpoint not found at {acmdm_ckpt_path}"
        
        acmdm_ckpt = torch.load(acmdm_ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = acmdm.load_state_dict(acmdm_ckpt['ema_acmdm'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        acmdm.eval()
        acmdm.to(device)
        _models_cache['acmdm'] = acmdm
        status_messages.append(f"✓ Loaded ACMDM from {acmdm_ckpt_path}")
        
        # Load LengthEstimator if needed
        length_estimator = None
        if use_length_estimator:
            status_messages.append("Loading LengthEstimator...")
            length_estimator = LengthEstimator(input_size=512, output_size=1)
            length_estimator_path = pjoin(checkpoints_dir, dataset_name, 'length_estimator', 'model', 'latest.tar')
            if os.path.exists(length_estimator_path):
                length_ckpt = torch.load(length_estimator_path, map_location='cpu')
                length_estimator.load_state_dict(length_ckpt['model'])
                length_estimator.eval()
                length_estimator.to(device)
                _models_cache['length_estimator'] = length_estimator
                status_messages.append(f"✓ Loaded LengthEstimator")
            else:
                status_messages.append(f"⚠ LengthEstimator not found, will use default length")
        
        # Load normalization stats
        status_messages.append("Loading normalization statistics...")
        after_mean = np.load(pjoin(checkpoints_dir, dataset_name, ae_name, 'AE_2D_Causal_Post_Mean.npy'))
        after_std = np.load(pjoin(checkpoints_dir, dataset_name, ae_name, 'AE_2D_Causal_Post_Std.npy'))
        joint_mean = np.load(f'utils/22x3_mean_std/{dataset_name}/22x3_mean.npy')
        joint_std = np.load(f'utils/22x3_mean_std/{dataset_name}/22x3_std.npy')
        eval_mean = np.load(f'utils/eval_mean_std/{dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{dataset_name}/eval_std.npy')
        
        _models_cache['stats'] = {
            'after_mean': after_mean,
            'after_std': after_std,
            'joint_mean': joint_mean,
            'joint_std': joint_std,
            'eval_mean': eval_mean,
            'eval_std': eval_std
        }
        
        _models_cache['loaded'] = True
        status_message = "\n".join(status_messages)
        return _models_cache, status_message
        
    except Exception as e:
        error_msg = f"Error loading models: {str(e)}"
        import traceback
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg


def estimate_motion_length(text: str, models_cache: dict) -> int:
    """Estimate motion length from text using LengthEstimator"""
    if models_cache['length_estimator'] is None:
        return None
    
    device = models_cache['device']
    acmdm = models_cache['acmdm']
    length_estimator = models_cache['length_estimator']
    
    with torch.no_grad():
        text_emb = acmdm.encode_text([text])
        pred_length = length_estimator(text_emb)
        pred_length = int(pred_length.item() * 4)
        pred_length = ((pred_length + 2) // 4) * 4
        pred_length = max(40, min(196, pred_length))
    return pred_length


def generate_motion_single(
    text: str,
    motion_length: Optional[int],
    cfg_scale: float,
    use_auto_length: bool,
    gpu_id: int,
    seed: int
) -> Tuple[Optional[str], str]:
    """
    Generate a single motion from text.
    Returns (video_path, status_message)
    """
    global _models_cache
    
    try:
        set_seed(seed)
        
        # Load models if not cached
        if not _models_cache['loaded']:
            models_cache, load_msg = load_models_cached(
                gpu_id=gpu_id,
                use_length_estimator=use_auto_length
            )
            if models_cache is None:
                return None, f"Failed to load models:\n{load_msg}"
        else:
            models_cache = _models_cache
        
        device = models_cache['device']
        ae = models_cache['ae']
        acmdm = models_cache['acmdm']
        stats = models_cache['stats']
        
        # Estimate length if needed
        if motion_length is None or (motion_length == 0 and use_auto_length):
            if use_auto_length and models_cache['length_estimator'] is not None:
                motion_length = estimate_motion_length(text, models_cache)
                status_msg = f"Estimated motion length: {motion_length} frames\n"
            else:
                motion_length = 120  # Default
                status_msg = f"Using default motion length: {motion_length} frames\n"
        else:
            # Round to multiple of 4
            motion_length = ((motion_length + 2) // 4) * 4
            status_msg = f"Using specified motion length: {motion_length} frames\n"
        
        status_msg += f"Generating motion for: '{text}'...\n"
        
        # Generate motion
        with torch.no_grad():
            latent_length = motion_length // 4
            m_lens = torch.tensor([latent_length], device=device)
            pred_latents = acmdm.generate([text], m_lens, cfg_scale)
            
            # Denormalize latents
            pred_latents_np = pred_latents.permute(0, 2, 3, 1).detach().cpu().numpy()
            pred_latents_np = pred_latents_np * stats['after_std'] + stats['after_mean']
            pred_latents_tensor = torch.from_numpy(pred_latents_np).to(device)
            
            # Decode through AE
            pred_motions = ae.decode(pred_latents_tensor.permute(0, 3, 1, 2))
            
            # Denormalize motions
            pred_motions_np = pred_motions.permute(0, 2, 3, 1).detach().cpu().numpy()
            if stats['joint_mean'].ndim == 1:
                pred_motions_np = pred_motions_np * stats['joint_std'][np.newaxis, np.newaxis, :, np.newaxis] + stats['joint_mean'][np.newaxis, np.newaxis, :, np.newaxis]
            else:
                pred_motions_np = pred_motions_np * stats['joint_std'][np.newaxis, ..., np.newaxis] + stats['joint_mean'][np.newaxis, ..., np.newaxis]
            
            # Back process to get RIC format, then recover joint positions
            from utils.motion_process import recover_from_ric
            
            motion = pred_motions_np[0]  # (22, 3, seq_len)
            motion = motion[:, :, :motion_length].transpose(2, 0, 1)  # (seq_len, 22, 3)
            ric_data = back_process(motion, is_mesh=False)
            ric_tensor = torch.from_numpy(ric_data).float()
            joints = recover_from_ric(ric_tensor, joints_num=22).numpy()  # (seq_len, 22, 3)
        
        # Create temporary file for video
        temp_dir = tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)
        video_path = pjoin(temp_dir, 'motion.mp4')
        
        # Generate video - plot_3d_motion returns the actual path (may be .mp4 or .gif)
        try:
            actual_video_path = plot_3d_motion(
                video_path,
                t2m_kinematic_chain,
                joints,
                title=text,
                fps=20,
                radius=4
            )
        except Exception as e:
            status_msg += f"\n⚠ Error in plot_3d_motion: {str(e)}"
            raise
        
        # Verify file exists - check both the returned path and original path
        if not os.path.exists(actual_video_path):
            # Check if file exists with original path
            if os.path.exists(video_path):
                actual_video_path = video_path
            # Check if it was saved as GIF instead
            elif os.path.exists(video_path.replace('.mp4', '.gif')):
                actual_video_path = video_path.replace('.mp4', '.gif')
            else:
                # List files in temp directory for debugging
                files_in_dir = os.listdir(temp_dir) if os.path.exists(temp_dir) else []
                error_msg = f"Video file not found. Expected: {actual_video_path}\n"
                error_msg += f"Files in temp directory: {files_in_dir}"
                status_msg += f"\n❌ {error_msg}"
                raise FileNotFoundError(error_msg)
        
        status_msg += f"\n✓ Motion generated successfully! Video saved."
        return actual_video_path, status_msg
        
    except Exception as e:
        error_msg = f"Error generating motion: {str(e)}"
        import traceback
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg


def generate_motion_batch(
    text_file_content: str,
    cfg_scale: float,
    use_auto_length: bool,
    gpu_id: int,
    seed: int
) -> Tuple[List[Optional[str]], str]:
    """
    Generate motions from a batch of text prompts.
    Returns (list_of_video_paths, status_message)
    """
    # Parse text file
    text_prompts = []
    for line in text_file_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if '#' in line:
            parts = line.split('#')
            text = parts[0].strip()
            length_str = parts[1].strip() if len(parts) > 1 else 'NA'
        else:
            text = line
            length_str = 'NA'
        
        if length_str.upper() == 'NA':
            motion_length = None if use_auto_length else 120
        else:
            try:
                motion_length = int(length_str)
                motion_length = ((motion_length + 2) // 4) * 4
            except:
                motion_length = None if use_auto_length else 120
        
        text_prompts.append((text, motion_length))
    
    if not text_prompts:
        return [], "No valid prompts found in file"
    
    status_msg = f"Processing {len(text_prompts)} prompts...\n"
    video_paths = []
    
    # Create a persistent directory for batch videos (not temp, so Gradio can access them)
    # Use a directory that Gradio can serve from
    batch_output_dir = pjoin('generation', 'batch_temp')
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Also ensure the parent directory exists
    os.makedirs('generation', exist_ok=True)
    
    for idx, (text, motion_length) in enumerate(text_prompts):
        try:
            video_path, gen_msg = generate_motion_single(
                text, motion_length, cfg_scale, use_auto_length, gpu_id, seed + idx
            )
            
            # If video was created, copy it to persistent location for batch gallery
            if video_path and os.path.exists(video_path):
                # Create a unique filename for this batch item
                # Get file extension from original path
                file_ext = os.path.splitext(video_path)[1] or '.mp4'
                batch_video_path = pjoin(batch_output_dir, f'motion_{idx:04d}{file_ext}')
                
                # Copy file to persistent location
                import shutil
                try:
                    shutil.copy2(video_path, batch_video_path)
                    
                    # Verify the copied file exists
                    if os.path.exists(batch_video_path):
                        # Use relative path for Gradio (works better in HuggingFace Spaces)
                        # Gradio can serve files from relative paths within the app directory
                        video_paths.append(batch_video_path)
                        file_size = os.path.getsize(batch_video_path)
                        status_msg += f"\n[{idx+1}/{len(text_prompts)}] ✓ {text[:50]}... - Video saved ({file_size/1024:.1f} KB)"
                    else:
                        status_msg += f"\n[{idx+1}/{len(text_prompts)}] ⚠ {text[:50]}... - Video copy failed (file not found after copy)"
                        video_paths.append(None)
                except Exception as copy_error:
                    status_msg += f"\n[{idx+1}/{len(text_prompts)}] ⚠ {text[:50]}... - Copy error: {str(copy_error)}"
                    video_paths.append(None)
            else:
                status_msg += f"\n[{idx+1}/{len(text_prompts)}] ❌ {text[:50]}... - Generation failed (no video path returned)"
                video_paths.append(None)
                
        except Exception as e:
            status_msg += f"\n[{idx+1}/{len(text_prompts)}] ❌ Error: {str(e)}"
            import traceback
            status_msg += f"\n   Traceback: {traceback.format_exc()[:200]}"
            video_paths.append(None)
    
    # Filter out None values and verify files exist - Gradio Gallery needs valid paths
    valid_video_paths = []
    for path in video_paths:
        if path is not None:
            # Use relative path (Gradio handles these better in cloud environments)
            # Verify file exists
            if os.path.exists(path):
                # Verify it's a video file
                if path.lower().endswith(('.mp4', '.gif', '.mov', '.avi')):
                    # Get file size for debugging
                    file_size = os.path.getsize(path)
                    if file_size > 0:  # Ensure file is not empty
                        valid_video_paths.append(path)  # Keep relative path
                        status_msg += f"\n  ✓ Verified: {os.path.basename(path)} ({file_size/1024:.1f} KB)"
                    else:
                        status_msg += f"\n  ⚠ Empty file: {path}"
                else:
                    status_msg += f"\n  ⚠ Skipping non-video file: {path}"
            else:
                # Try absolute path as fallback
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    valid_video_paths.append(abs_path)
                    status_msg += f"\n  ✓ Found via absolute path: {abs_path}"
                else:
                    status_msg += f"\n  ⚠ File not found (relative: {path}, absolute: {abs_path})"
    
    if len(valid_video_paths) == 0:
        status_msg += "\n\n❌ No videos were successfully generated for the gallery."
        status_msg += "\n💡 Check the error messages above for each prompt."
        # Return empty list explicitly
        return [], status_msg
    else:
        status_msg += f"\n\n✅ Successfully generated {len(valid_video_paths)}/{len(text_prompts)} motions."
        status_msg += f"\n📁 Videos saved in: {batch_output_dir}"
        status_msg += f"\n📋 Gallery will display {len(valid_video_paths)} video(s)."
        
        # Debug: Show first few paths
        status_msg += f"\n\n🎬 First few video paths:\n"
        for i, p in enumerate(valid_video_paths[:3]):
            abs_p = os.path.abspath(p) if not os.path.isabs(p) else p
            exists = "✓" if os.path.exists(p) else "✗"
            status_msg += f"  {exists} [{i+1}] {p} (exists: {os.path.exists(p)})\n"
        if len(valid_video_paths) > 3:
            status_msg += f"  ... and {len(valid_video_paths) - 3} more\n"
    
    # Return list of file paths (Gradio Gallery expects list of strings)
    # Ensure all paths are valid and accessible
    final_paths = []
    for p in valid_video_paths:
        if os.path.exists(p):
            # Normalize path (use forward slashes for cross-platform compatibility)
            normalized_path = p.replace('\\', '/')
            final_paths.append(normalized_path)
        else:
            status_msg += f"\n⚠ Warning: Path not found when returning: {p}"
    
    if len(final_paths) != len(valid_video_paths):
        status_msg += f"\n⚠ Some paths were filtered out. Returning {len(final_paths)}/{len(valid_video_paths)} videos."
    
    return final_paths, status_msg


# Gradio Interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="ACMDM Motion Generation", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎭 ACMDM Motion Generation
        
        Generate human motion from text descriptions using the ACMDM (Absolute Coordinates Make Motion Generation Easy) model.
        
        **How to use:**
        1. Enter a text description of the motion you want to generate
        2. Adjust motion length (or use auto-estimate)
        3. Click "Generate Motion" to create the animation
        4. View and download the generated video
        
        **Example prompts:**
        - "A person is running on a treadmill."
        - "Someone is doing jumping jacks."
        - "A person walks forward and then turns around."
        """)
        
        with gr.Tabs():
            # Single Generation Tab
            with gr.Tab("Single Motion Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Motion Description",
                            placeholder="A person is running on a treadmill.",
                            lines=3,
                            value="A person is running on a treadmill."
                        )
                        
                        with gr.Row():
                            motion_length = gr.Slider(
                                label="Motion Length (frames)",
                                minimum=40,
                                maximum=196,
                                value=120,
                                step=4,
                                info="Will be rounded to nearest multiple of 4"
                            )
                            use_auto_length = gr.Checkbox(
                                label="Auto-estimate length",
                                value=False,
                                info="Use length estimator (ignores manual length if checked)"
                            )
                        
                        cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5,
                            info="Classifier-free guidance scale (higher = more aligned with text)"
                        )
                        
                        with gr.Row():
                            gpu_id = gr.Number(
                                label="GPU ID",
                                value=0,
                                precision=0,
                                info="Use -1 for CPU"
                            )
                            seed = gr.Number(
                                label="Random Seed",
                                value=3407,
                                precision=0
                            )
                        
                        generate_btn = gr.Button("Generate Motion", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        video_output = gr.Video(
                            label="Generated Motion",
                            format="mp4"
                        )
                        status_output = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False
                        )
                
                # Update model status after generation
                def generate_and_update_status(text, motion_length, cfg_scale, use_auto_length, gpu_id, seed):
                    video_path, status_msg = generate_motion_single(
                        text, motion_length, cfg_scale, use_auto_length, gpu_id, seed
                    )
                    # Return video and status, plus trigger status update
                    return video_path, status_msg
                
                generate_btn.click(
                    fn=generate_and_update_status,
                    inputs=[text_input, motion_length, cfg_scale, use_auto_length, gpu_id, seed],
                    outputs=[video_output, status_output]
                )
            
            # Batch Generation Tab
            with gr.Tab("Batch Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_text_input = gr.Textbox(
                            label="Text Prompts (one per line, format: text#length or text#NA)",
                            placeholder="A person is running on a treadmill.#120\nSomeone is doing jumping jacks.#NA",
                            lines=10,
                            info="Each line: 'text#length' or 'text#NA' for auto-estimate"
                        )
                        
                        batch_cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5
                        )
                        
                        batch_use_auto_length = gr.Checkbox(
                            label="Auto-estimate length for NA",
                            value=True
                        )
                        
                        batch_gpu_id = gr.Number(
                            label="GPU ID",
                            value=0,
                            precision=0
                        )
                        
                        batch_seed = gr.Number(
                            label="Random Seed",
                            value=3407,
                            precision=0
                        )
                        
                        batch_generate_btn = gr.Button("Generate Batch", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        batch_status_output = gr.Textbox(
                            label="Batch Status",
                            lines=15,
                            interactive=False
                        )
                        batch_video_gallery = gr.Gallery(
                            label="Generated Motions",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto",
                            type="filepath"  # Explicitly specify filepath type
                        )
                
                batch_generate_btn.click(
                    fn=generate_motion_batch,
                    inputs=[batch_text_input, batch_cfg_scale, batch_use_auto_length, batch_gpu_id, batch_seed],
                    outputs=[batch_video_gallery, batch_status_output]
                )
            
            # Model Management Tab
            with gr.Tab("Model Management"):
                with gr.Row():
                    with gr.Column():
                        model_status = gr.Textbox(
                            label="Model Status",
                            lines=15,
                            interactive=False,
                            value="⏳ Models not loaded yet. They will be loaded automatically on first generation."
                        )
                        with gr.Row():
                            refresh_status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                            reload_models_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                        
                        gr.Markdown("""
                        **Model Configuration:**
                        - Model Name: `ACMDM_Flow_S_PatchSize22`
                        - AE Name: `AE_2D_Causal`
                        - Dataset: `t2m`
                        - Checkpoints Directory: `./checkpoints`
                        """)
                
                def check_model_status():
                    """Check and display current model status"""
                    global _models_cache
                    if _models_cache['loaded']:
                        device = _models_cache['device']
                        status_lines = [
                            "✓ MODELS LOADED AND READY",
                            "=" * 50,
                            f"📱 Device: {device}",
                            f"💾 CUDA Available: {torch.cuda.is_available()}",
                        ]
                        
                        if device.type == 'cuda':
                            status_lines.append(f"🎮 GPU: {torch.cuda.get_device_name(device.index if device.index is not None else 0)}")
                            status_lines.append(f"💾 GPU Memory: {torch.cuda.get_device_properties(device.index if device.index is not None else 0).total_memory / 1e9:.2f} GB")
                        
                        status_lines.extend([
                            "",
                            "📦 Loaded Models:",
                            "  ✓ Autoencoder (AE_2D_Causal)",
                            "  ✓ ACMDM Diffusion Model",
                        ])
                        
                        if _models_cache['length_estimator'] is not None:
                            status_lines.append("  ✓ Length Estimator")
                        else:
                            status_lines.append("  ⚠ Length Estimator (not loaded)")
                        
                        status_lines.extend([
                            "",
                            "📊 Statistics Loaded:",
                            "  ✓ Post-AE Mean/Std",
                            "  ✓ Joint Mean/Std",
                            "  ✓ Eval Mean/Std",
                            "",
                            "✨ Models are cached and ready for generation!",
                            "💡 Tip: Use 'Clear Cache' to force reload on next generation."
                        ])
                        
                        return "\n".join(status_lines)
                    else:
                        return (
                            "⏳ Models not loaded yet. They will be loaded automatically on first generation.\n"
                            "📝 To load models now, go to 'Single Motion Generation' tab and click 'Generate Motion'.\n"
                            "⏱️ First generation will take 30-60 seconds (model loading time).\n"
                            "⚡ Subsequent generations will be much faster (5-15 seconds)."
                        )
                
                def reload_models():
                    """Clear model cache"""
                    global _models_cache
                    _models_cache['loaded'] = False
                    _models_cache['ae'] = None
                    _models_cache['acmdm'] = None
                    _models_cache['length_estimator'] = None
                    _models_cache['stats'] = None
                    _models_cache['device'] = None
                    return (
                        "🗑️ Model cache cleared. Models will be automatically reloaded on your next generation request.\n"
                        "💡 Click 'Refresh Status' after generating to see updated status."
                    )
                
                # Button callbacks
                refresh_status_btn.click(
                    fn=check_model_status,
                    outputs=model_status
                )
                
                reload_models_btn.click(
                    fn=reload_models,
                    outputs=model_status
                )
                
                # Update status on tab load
                app.load(
                    fn=check_model_status,
                    outputs=model_status
                )
        
        gr.Markdown("""
        ---
        **Note:** First generation may take longer as models need to be loaded. Subsequent generations will be faster.
        
        **Tips:**
        - Use descriptive text prompts for better results
        - Adjust CFG scale: higher values (3-5) for more text alignment, lower values (1-2) for more diversity
        - Motion length should be a multiple of 4 (automatically rounded)
        - For batch processing, use the format: `text description#length` or `text description#NA`
        """)
    
    return app


if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    
    # Launch with sharing option
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public link (requires ngrok)
        show_error=True
    )

