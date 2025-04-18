import pandas as pd
import numpy as np
import math
import cv2
from moviepy import VideoFileClip, concatenate_videoclips

class VideoClipper:
    def __init__(self, video_file, h5_file):
        self.video_file = video_file
        self.h5_file = h5_file
        self.df = pd.read_hdf(h5_file)
        self.fps = self.get_video_fps(video_file)
        self.speeds = []
        self.interesting_frames = []

    def get_video_fps(self, video_file):
        """Get the fps of the video"""
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_file}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return fps

    def calculate_speed(self, bodyparts):
        """Calculates the average speed of specified body parts."""
        speeds = []
        individuals = 'animal0' # hardcoded, assumpution is that only 1 pet in the video, therefore the coordinates of the first animal are used
        scorer = self.df.columns.get_level_values('scorer')[0]  # The column name of the first level of df
        
        for i in range(1, len(self.df)):  # Start from the second frame
            frame1 = self.df.iloc[i-1]
            frame2 = self.df.iloc[i]
            total_displacement = 0
            valid_parts = 0  # Count body parts with valid data

            for part in bodyparts:
                x1 = frame1[(scorer, individuals, part, 'x')]
                y1 = frame1[(scorer, individuals, part, 'y')]
                x2 = frame2[(scorer, individuals, part, 'x')]
                y2 = frame2[(scorer, individuals, part, 'y')]

                # Check for NaN values (missing data)
                if not np.any(np.isnan([x1, y1, x2, y2])):
                    displacement = math.dist((x1, y1), (x2, y2))
                    total_displacement += displacement
                    valid_parts += 1

            if valid_parts > 0:
                avg_displacement = total_displacement / valid_parts
            else:
                avg_displacement = 0  # Handle the case where no parts have valid data

            speed = avg_displacement / (1 / self.fps)
            speeds.append(speed)

        self.speeds = speeds

    def identify_high_speed_frames(self, std_multiplier=2, window_size=5):
        """Identifies interesting frames using an adaptive threshold (mean + std)."""
        rolling_avg_speeds = pd.Series(self.speeds).rolling(window=window_size).mean().fillna(0).tolist()
        mean_speed = np.mean(rolling_avg_speeds)
        std_speed = np.std(rolling_avg_speeds)
        speed_threshold = mean_speed + std_multiplier * std_speed

        interesting_frames = []
        for i, speed in enumerate(rolling_avg_speeds):
            if speed > speed_threshold:
                frame_number = i + 1
                interesting_frames.append((frame_number, f"Speed: {speed:.2f}"))
        self.interesting_frames = interesting_frames

    def clip_video_segments(self, buffer_duration=0.2):
        """Creates and merges clips to avoid repetition."""
        clips = []
        merged_intervals = []  # Store merged time intervals

        for frame_num, reason in self.interesting_frames:
            start_time = (frame_num - 1) / self.fps - buffer_duration
            end_time = frame_num / self.fps + buffer_duration
            start_time = max(0, start_time)  # Ensure start_time is not negative

            current_interval = (start_time, end_time)

            # Check for overlap with existing intervals
            merged = False
            for i, (existing_start, existing_end) in enumerate(merged_intervals):
                if start_time <= existing_end and end_time >= existing_start:  # Overlap detected
                    # Merge the intervals
                    merged_intervals[i] = (min(start_time, existing_start), max(end_time, existing_end))
                    merged = True
                    break

            if not merged:
                merged_intervals.append(current_interval)

        # Create clips from the merged intervals
        for start_time, end_time in merged_intervals:
            try:
                clip = VideoFileClip(self.video_file).subclipped(start_time, end_time)
                clips.append(clip)
            except Exception as e:
                print(f"Error creating clip from {start_time:.2f} to {end_time:.2f}: {e}")

        # Concatenate the clips
        if clips:
            final_clip = concatenate_videoclips(clips)
            return final_clip
        else:
            print("No valid clips to concatenate.")
            return None

    def analyze_and_clip(self, output_filename, buffer_duration=0.2, std_multiplier=2, window_size=5):
        """Main method to perform analysis and create video clips."""
        all_bodyparts = self.df.columns.get_level_values('bodyparts').unique()
        self.calculate_speed(all_bodyparts)
        self.identify_high_speed_frames(std_multiplier=std_multiplier, window_size=window_size)
        final_clip = self.clip_video_segments(buffer_duration=buffer_duration)

        if final_clip:
            output_path = f"output/{output_filename}"
            final_clip.write_videofile(output_path, codec="libx264")
        else:
            print("No video created.")

if __name__ == "__main__":
    
    video_file = r"3191251-uhd_4096_2160_25fps/3191251-uhd_4096_2160_25fps.mp4"
    h5_file = r"3191251-uhd_4096_2160_25fps/3191251-uhd_4096_2160_25fps_superanimal_quadruped_snapshot-fasterrcnn_resnet50_fpn_v2-004_snapshot-hrnet_w32-004.h5"

    clipper = VideoClipper(video_file, h5_file)
    clipper.analyze_and_clip("interesting_segments_clip.mp4", buffer_duration=0.2, std_multiplier=0.5, window_size=5)