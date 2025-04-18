import torch
import clip
from PIL import Image
import numpy as np
import cv2
from os import path, makedirs
from .ImageOverlay import ImageOverlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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

def generate_image_overlays(extracted_frames, model, preprocess, device, caption_options, font, color=(255, 255, 255, 255), bg_tint_color=(0, 0, 0), bg_transparency=0, draw_shadow=True, animate_text=False, animation_interval=5, animation_offset=0.01):
    """Generates text overlays for each frame using CLIP and ImageOverlay class."""

    # Pre-encode caption options
    text = clip.tokenize(caption_options).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Initialize array to store cumulative similarity scores
    cumulative_similarity = np.zeros(len(caption_options))

    # First pass: calculate cumulative similarity scores across all frames
    for frame_path in extracted_frames:
        try:
            image = Image.open(frame_path).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze().cpu().numpy()
            cumulative_similarity += similarity
        except Exception as e:
            print(f"Error processing frame {frame_path} for similarity: {e}")

    # Find the top 2 captions based on cumulative scores
    top_2_indices = np.argsort(cumulative_similarity)[-2:][::-1]  # Get indices of top 2 scores
    best_captions = [caption_options[i] for i in top_2_indices]

    overlayed_images = []

    # Define base text positions
    base_text_positions = [
        (0.2, 0.15, best_captions[0]),  # First caption base position
        (0.8, 0.25, best_captions[1])   # Second caption base position
    ]

    # Second pass: apply the captions to all frames
    for i, frame_path in enumerate(extracted_frames): # Added index 'i'
        try:
            overlay = ImageOverlay(frame_path, font, bg_tint_color, bg_transparency, color, draw_shadow)
            if overlay.load_image():
                current_text_positions = base_text_positions
                if animate_text:
                    # Calculate offset based on frame index and interval
                    # Simple alternating horizontal offset for demonstration
                    offset_factor = ((i // animation_interval) % 2) * 2 - 1 # Alternates between -1 and 1
                    x_offset = animation_offset * offset_factor
                    # Apply offset to base positions
                    current_text_positions = [
                        (pos[0] + x_offset, pos[1], pos[2]) # Apply horizontal offset
                        for pos in base_text_positions
                    ]

                overlay.overlay_text(current_text_positions) # Use potentially offset positions
                overlayed_images.append(path.splitext(frame_path)[0] + "_overlay.png")
            else:
                print(f"Error loading or overlaying image: {frame_path}")
                overlayed_images.append(frame_path)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            overlayed_images.append(frame_path)

    return overlayed_images

def create_video_from_images(image_paths, output_video="output/output.mp4", fps=25):
    """Creates a video from a list of images."""
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(output_video, fps=fps, codec="libx264")

def pipeline(video_path, caption_options, font, output_video="output/output.mp4",bg_tint_color=(0, 0, 0), bg_transparency=0, color=(255, 255, 255, 255), draw_shadow=True, animate_text=False, animation_interval=10, animation_offset=0.05):
    """Main function to run CLIP-based image overlay and video creation."""

    # 1. Load CLIP model
    model, preprocess, device = load_clip_model()

    # 2. Extract frames
    fps, extracted_frames = extract_frames(video_path)

    # 3. Generate image overlays
    overlayed_images = generate_image_overlays(
        extracted_frames, model, preprocess, device, caption_options, font,
        bg_tint_color=bg_tint_color, bg_transparency=bg_transparency, color=color, draw_shadow=draw_shadow,
        animate_text=animate_text, animation_interval=animation_interval, animation_offset=animation_offset
    )

    # 4. Create video from images
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