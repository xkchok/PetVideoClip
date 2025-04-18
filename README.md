# Generating Short Clips for Pet Videos
Automatically clip pet videos based on their activities/movement and add text overlay on the video

## Setup
The project environment is managed using uv.  
Install here: https://github.com/astral-sh/uv  
```bash 
$ uv sync
```
And you should be good to go!

## Usage
```bash
$ uv run main.py
```
If you'd like to execute the files under src/ as a script, 
```bash
$ uv run -m src.text_overlay
```
Check src files for more configurations of parameters

## Workflow
1. An HDF5 file is generated using the superanimal quadruped model from DeepLabCut. Refer to [here](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_DEMO_SuperAnimal.ipynb).  
You can perform an analysis on the h5 file using pandas. E.g.,
```python
import pandas as pd

h5_file_path = 'path/to/file'
df = pd.read_hdf(h5_file_path)
```

---

2. The video is clipped based on the data from the HDF5 file. 
  * Calculates the speed of the animal's movement between consecutive frames.
  * Uses a rolling average to smooth the speed data
  * Sets a threshold based on mean speed + (standard deviation × multiplier) 
  * Identifies frames where speed exceeds this threshold
  * Stores these frames as "interesting"
  * For each interesting frame:
    * Creates a time window (with buffer before/after)
    * Merges overlapping time windows to avoid repetition
    * Creates video clips for each merged time window
    * Concatenates all clips into a single video

---

3. Text overlays are added to the clipped video using OpenAI CLIP
  * For each frame:
    * CLIP model analyzes the frame content
    * Calculates similarity scores with predefined captions
  * CLIP model selects the most relevant captions based on the cumulative similarity scores across all frames
  * Text overlays are applied to frames with:
    * Customizable font and color
    * Optional text animation
    * Optional background tinting
    * Optional shadow effects
  * Final video is created from the overlayed frames

  ---