from moviepy.editor import VideoFileClip, AudioFileClip


def replace_audio_in_video(video_path, audio_path, output_path):
    # Load the video file and remove its audio
    video_clip = VideoFileClip('Data/CutVideo.mp4').without_audio()

    # Load the new audio file
    audio_clip = AudioFileClip('Data/LoweredAndMuted.wav')

    # Set the processed audio file as the new audio clip
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file
    final_clip.write_videofile('Data/VideoProcessedLoweredAndMuted.MP4', codec='libx264', audio_codec='aac')


# Replace audio
replace_audio_in_video('Data/CutVideo.mp4', 'Data/LoweredAndMuted.wav', 'Data/VideoProcessedLoweredAndMuted.MP4')
