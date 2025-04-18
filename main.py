from src import clip_video, text_overlay
import os

def main():
    video_file = "3191251-uhd_4096_2160_25fps/3191251-uhd_4096_2160_25fps.mp4"
    h5_file = "3191251-uhd_4096_2160_25fps/3191251-uhd_4096_2160_25fps_superanimal_quadruped_snapshot-fasterrcnn_resnet50_fpn_v2-004_snapshot-hrnet_w32-004.h5"
    clipped_video_file = "interesting_segments_clip.mp4"

    # Clip
    clipper = clip_video.VideoClipper(video_file, h5_file)
    clipper.analyze_and_clip(clipped_video_file, buffer_duration=0.2, std_multiplier=0.5, window_size=5)

    # Text Overlay
    clipped_path = os.path.join("output", clipped_video_file)
    output_video = "output/overlayed_video.mp4"
    caption_options = [
        "Cute!",
        "Playing!",
        "So adorable!",
        "Having fun!",
        "A happy moment.",
        "Look at that activity!"
    ]
    font = "fonts/LoveDays-2v7Oe.ttf"
    text_overlay.pipeline(
        clipped_path,
        caption_options,
        font,
        output_video,
        animate_text=True,
    )

if __name__ == "__main__":
    main()
