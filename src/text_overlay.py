import torch
import clip
from PIL import Image
import numpy as np
import cv2
from os import path, makedirs, process_cpu_count
from .ImageOverlay import ImageOverlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from multiprocessing import Pool
from tqdm import tqdm

def extract_frames(video_path, output_dir="frames", frame_interval=1):
    """Extracts frames from the video at a specified interval."""
    if not path.exists(output_dir):
        makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    extracted_frames = []  # Store paths to extracted frames
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_interval == 0:
            frame_path = path.join(output_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)

    cap.release()
    return fps, extracted_frames

def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads the CLIP model and tokenizer."""
    model, preprocess = clip.load("ViT-B/32", device=device) # can try "ViT-L/14" 
    return model, preprocess, device

def process_single_frame(args):
    """Process a single frame with text overlay
    args: tuple containing (frame_index, frame_path, overlay_params)
    """
    i, frame_path, params = args
    try:
        overlay = ImageOverlay(
            frame_path, 
            params['font'], 
            params['bg_tint_color'], 
            params['bg_transparency'], 
            params['color'], 
            params['draw_shadow']
        )
        
        if not overlay.load_image():
            print(f"Failed to load image: {frame_path}")
            return frame_path

        current_text_positions = params['base_text_positions']
        
        if params['animate_text']:
            offset_factor = ((i // params['animation_interval']) % 2) * 2 - 1
            x_offset = params['animation_offset'] * offset_factor
            current_text_positions = [
                (pos[0] + x_offset, pos[1], pos[2])
                for pos in params['base_text_positions']
            ]
            
        overlay.overlay_text(current_text_positions)
        return path.splitext(frame_path)[0] + "_overlay.png"
        
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}")
        return frame_path

def process_clip_batch(frame_paths, model, preprocess, device, batch_size=16):
    """Process a batch of frames through CLIP"""
    batch_features = []
    
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        try:
            # Load and preprocess batch of images
            batch_images = torch.stack([
                preprocess(Image.open(path).convert("RGB"))
                for path in batch_paths
            ]).to(device)

            # Get features for batch
            with torch.no_grad():
                features = model.encode_image(batch_images)
                features /= features.norm(dim=-1, keepdim=True)
                batch_features.append(features)

        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            # Return zero features for failed batch
            zero_features = torch.zeros((len(batch_paths), model.visual.output_dim)).to(device)
            batch_features.append(zero_features)

    # Concatenate all batch features
    return torch.cat(batch_features, dim=0)

def generate_image_overlays(extracted_frames, model, preprocess, device, caption_options, font, 
                          color=(255, 255, 255, 255), bg_tint_color=(0, 0, 0), bg_transparency=0, 
                          draw_shadow=True, animate_text=False, animation_interval=5, 
                          animation_offset=0.01, clip_batch_size=16, num_processes=None,
                          num_captions=4):
    """Generates text overlays for each frame using CLIP and ImageOverlay class."""
    
    if num_processes is None:
        num_processes = max(1, process_cpu_count() - 1)
    
    num_captions = max(1, min(4, num_captions))  # Clamp between 1 and 4
    
    print("Processing text features...")
    text = clip.tokenize(caption_options).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Processing image features...")
    image_features = process_clip_batch(
        extracted_frames, 
        model, 
        preprocess, 
        device, 
        batch_size=clip_batch_size
    )

    similarities = (image_features @ text_features.T).cpu().numpy()
    cumulative_similarity = similarities.sum(axis=0)

    top_n_indices = np.argsort(cumulative_similarity)[-num_captions:][::-1]
    best_captions = [caption_options[i] for i in top_n_indices]

    # Define base text positions based on number of captions
    if num_captions == 1:
        base_text_positions = [
            (0.5, 0.85, best_captions[0])  # Bottom middle for single caption
        ]
    else:
        # Predefined positions for 2-4 captions
        positions = [
            (0.2, 0.15),  # Top left
            (0.8, 0.25),  # Top right
            (0.3, 0.85),  # Bottom left
            (0.7, 0.75)   # Bottom right
        ]
        base_text_positions = [
            (x, y, caption) for (x, y), caption in zip(positions[:num_captions], best_captions)
        ]

    # Prepare parameters for parallel processing
    overlay_params = {
        'font': font,
        'bg_tint_color': bg_tint_color,
        'bg_transparency': bg_transparency,
        'color': color,
        'draw_shadow': draw_shadow,
        'animate_text': animate_text,
        'animation_interval': animation_interval,
        'animation_offset': animation_offset,
        'base_text_positions': base_text_positions
    }

    # Prepare frame data with indices
    frame_data = [(i, frame_path, overlay_params) 
                  for i, frame_path in enumerate(extracted_frames)]

    # Process frames in parallel
    print("Applying text overlays...")
    with Pool(processes=num_processes) as pool:
        overlayed_images = list(tqdm(
            pool.imap(process_single_frame, frame_data),
            total=len(frame_data),
            desc="Processing frames"
        ))

    return overlayed_images

def create_video_from_images(image_paths, output_video="output/output.mp4", fps=25):
    """Creates a video from a list of images."""
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(output_video, fps=fps, codec="libx264")

def pipeline(video_path, caption_options, font, output_video="output/output.mp4",
            bg_tint_color=(0, 0, 0), bg_transparency=0, color=(255, 255, 255, 255),
            draw_shadow=True, animate_text=False, animation_interval=10,
            animation_offset=0.01, clip_batch_size=16, num_processes=None):
    """Main function to run CLIP-based image overlay and video creation."""
    
    # 1. Load CLIP model
    model, preprocess, device = load_clip_model()

    # 2. Extract frames
    fps, extracted_frames = extract_frames(video_path)

    # 3. Generate image overlays with batching and progress bars
    overlayed_images = generate_image_overlays(
        extracted_frames, model, preprocess, device, caption_options, font,
        bg_tint_color=bg_tint_color, bg_transparency=bg_transparency,
        color=color, draw_shadow=draw_shadow, animate_text=animate_text,
        animation_interval=animation_interval, animation_offset=animation_offset,
        clip_batch_size=clip_batch_size, num_processes=num_processes
    )

    # 4. Create video from images
    print("Creating final video...")
    create_video_from_images(overlayed_images, output_video=output_video, fps=fps)

if __name__ == "__main__":
    video_path = r"output/interesting_segments_clip.mp4"
    output_video = r"output/overlayed_video.mp4"

    caption_options = [
        "Cute!",
        "Playing!",
        "So adorable!",
        "Having fun!",
        "A happy moment.",
        "Look at that activity!"
        # Add more
    ]

    font = r"fonts/LoveDays-2v7Oe.ttf"
    bg_tint_color = (0, 0, 0)
    bg_transparency = 0
    alpha = 1
    alpha = int(alpha * 255)
    text_color = (254, 153, 0, alpha) # RGBA

    animate = True 
    interval = 10  # Change position every 10 frames
    offset = 0.005 # Offset amount (as ratio of image width)

    pipeline(
        video_path,
        caption_options,
        font,
        output_video,
        bg_tint_color=bg_tint_color,
        bg_transparency=bg_transparency,
        color=text_color,
        draw_shadow=False,
        animate_text=animate,         
        animation_interval=interval,  
        animation_offset=offset
    )