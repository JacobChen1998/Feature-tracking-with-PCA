import imageio
from tkinter import filedialog
import os
import cv2
folder_path = filedialog.askdirectory()
FPS = 30

file = os.listdir(folder_path)
# file.sort()
# file.sort(key = lambda x: int(x[:-4]))
file = sorted( file,
                        key = lambda x: os.path.getmtime(os.path.join(folder_path, x))
                        )

frames = []
for frame_index in range(len(file)):
    print(folder_path+file[frame_index])
    frames.append(cv2.cvtColor(cv2.imread(folder_path+"/"+file[frame_index]), cv2.COLOR_BGR2RGB))
    
with imageio.get_writer("demo.gif", mode="I",fps=FPS) as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame)

print("\nFinished")