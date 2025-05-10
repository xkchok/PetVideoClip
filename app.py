import streamlit as st
import os
from src import clip_video, text_overlay
import sys
from io import StringIO
import contextlib
from PIL import Image, ImageDraw, ImageFont
import io

@contextlib.contextmanager
def capture_stdout():
    """Capture stdout and yield a StringIO object containing the output"""
    stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def main():
    # Create necessary directories at startup
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    st.title("🎥 Pet Video Clipper")

    st.write("---")
    
    # File uploaders
    st.header("📁 1. Upload Files")
    video_file = st.file_uploader("Upload Video File", type=['mp4'])
    h5_file = st.file_uploader("Upload H5 File", type=['h5'])
    
    # Save uploaded files temporarily if they exist
    temp_video_path = None
    temp_h5_path = None
    if video_file is not None:
        temp_video_path = f"temp/{video_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.getvalue())
    
    if h5_file is not None:
        temp_h5_path = f"temp/{h5_file.name}"
        with open(temp_h5_path, "wb") as f:
            f.write(h5_file.getvalue())

    st.write("---")
    
    # Clipping parameters
    st.header("✂️ 2. Clipping Parameters")
    clipped_video_name = st.text_input("Clipped Video Filename", "interesting_segments_clip.mp4")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        buffer_duration = st.number_input(
            "Buffer Duration (seconds)", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.2, 
            step=0.1,
            help="Controls the time window around interesting frames. A larger buffer will include more context before and after each detected moment."
        )
    
    with col2:
        std_multiplier = st.number_input(
            "Standard Deviation Multiplier", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.5, 
            step=0.1,
            help="Adjusts the sensitivity of the threshold. Higher values make detection more selective, while lower values will detect more moments."
        )
    
    with col3:
        window_size = st.number_input(
            "Window Size", 
            min_value=3, 
            max_value=15, 
            value=5, 
            step=2,
            help="Sets the rolling average window for smoothing speed data. Larger windows create smoother transitions but might miss quick movements."
        )
    
    st.write("---")
    
    # Text overlay parameters
    st.header("✨ 3. Text Overlay Parameters")
    output_video_name = st.text_input("Output Video Filename", "overlayed_video.mp4")
    
    # Caption management
    st.subheader("💭 Captions")
    num_captions = st.number_input("Number of Captions", min_value=1, max_value=4, value=1)
    
    default_captions = [
        "Cute!", "Playing!", "So adorable!", 
        "Having fun!", "A happy moment.", "Look at that activity!"
    ]
    
    # Initialize session state for captions if not exists
    if 'captions' not in st.session_state:
        st.session_state.captions = default_captions
    
    # Add new caption
    new_caption = st.text_input("Add New Caption")
    if st.button("Add Caption") and new_caption:
        st.session_state.captions.append(new_caption)
    
    # Display and manage existing captions
    st.write("Current Captions:")
    for i, caption in enumerate(st.session_state.captions):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(caption)
        with col2:
            if st.button(f"Remove", key=f"remove_{i}"):
                st.session_state.captions.pop(i)
                st.rerun()
    
    # Font selection with preview
    st.subheader("🔤 Font Settings")
    available_fonts = [f for f in os.listdir("fonts") if f.endswith(('.ttf', '.otf'))]
    
    # Create two columns for font selection and preview
    font_col1, font_col2 = st.columns([1, 2])
    
    with font_col1:
        selected_font = st.selectbox("Select Font", available_fonts)
        font_path = f"fonts/{selected_font}"
    
    with font_col2:
        # Create a preview of the selected font
        preview_text = "Preview Text"
        try:
            # Load and render the font
            font_size = 40
            font = ImageFont.truetype(font_path, font_size)
            
            # Draw the text
            img = Image.new('RGB', (400, 100), color='white')
            d = ImageDraw.Draw(img)
            d.text((10, 30), preview_text, font=font, fill='black')
            
            # Convert to bytes for display
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            # Display the preview
            st.image(buf, caption="Font Preview")
            
        except Exception as e:
            st.error(f"Could not preview font: {str(e)}")
    
    # Color and style settings
    st.subheader("🎨 Style Settings")
    col1, col2 = st.columns(2)
    with col1:
        font_color = st.color_picker("Font Color", "#FFFFFF")
        font_color = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    with col2:
        bg_color = st.color_picker("Background Color", "#000000")
        bg_color = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    bg_transparency = st.slider("Background Transparency", 0, 100, 0)
    draw_shadow = st.checkbox("Draw Shadow", True)
    animate_text = st.checkbox("Animate Text", False)
    
    # Process button
    if st.button("Process Video"):
        if video_file is None or h5_file is None:
            st.error("Please upload both video and H5 files.")
            return
        
        try:
            # Use forward slashes for paths
            clipped_path = f"output/{clipped_video_name}"
            output_path = f"output/{output_video_name}"
            
            with st.status("Processing video...", expanded=True) as status:
                # Create a placeholder for terminal output
                terminal_output = st.empty()
                output_text = ""
                
                # Clip video
                status.update(label="Step 1: Analyzing and clipping interesting segments...", state="running")
                with capture_stdout() as output:
                    clipper = clip_video.VideoClipper(temp_video_path, temp_h5_path)
                    clipper.analyze_and_clip(
                        clipped_path,
                        buffer_duration=buffer_duration,
                        std_multiplier=std_multiplier,
                        window_size=window_size
                    )
                    # Accumulate output
                    output_text += output.getvalue()
                    terminal_output.code(output_text)
                
                status.update(label="Step 2: Adding text overlays...", state="running")
                
                # Add text overlay
                with capture_stdout() as output:
                    text_overlay.pipeline(
                        clipped_path,
                        st.session_state.captions,
                        font_path,
                        output_path,
                        color=font_color,
                        bg_tint_color=bg_color,
                        bg_transparency=bg_transparency,
                        draw_shadow=draw_shadow,
                        animate_text=animate_text,
                        num_captions=num_captions
                    )
                    # Accumulate and update output
                    output_text += output.getvalue()
                    terminal_output.code(output_text)
                
                status.update(label="Video processing completed!", state="complete")
            
            st.success("Video processing completed successfully!")
            
            # Show download button for the processed video
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name=output_video_name,
                    mime="video/mp4"
                )
            
            # Cleanup temporary files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 