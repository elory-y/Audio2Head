import os
import glob
from moviepy.editor import *

def mp4_wav(mp4):
    change_name =mp4.replace("mp4", "wav")
    video = VideoFileClip(mp4)
    audio = video.audio
    audio.write_audiofile(change_name)



if __name__ == "__main__":
    mp4_path = glob.glob("/home/caopu/workspace/Audio2Head/test_data/*.mp4")
    save_path = r"D:\zqy\wav"
    mp4_wav("/home/caopu/workspace/Audio2Head/test_data/5.mp4")
    # for mp4 in mp4_path:
    #     mp4_wav(mp4)
