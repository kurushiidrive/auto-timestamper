import tkinter
from tkinter import filedialog
from tkinter import scrolledtext

import cv2 as cv
import numpy as np
import sys
import os               # for pyinstaller onefile
from datetime import timedelta
from skimage.metrics import structural_similarity as compare_ssim
import pytube
import threading

#import pafy
#import youtube_dl


# https://stackoverflow.com/questions/51060894/adding-a-data-file-in-pyinstaller-using-the-onefile-option
def resource_path(relative_path):
    ''' Get absolute path to resource '''
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

debug = True            # verbose debug output flag
pack = False            # create the debugOutput window only once
thrd = None             # thread for executing divide(), which processes the video
execution = False       # flag for whether video is currently being processed
cancel = False          # flag for whether video processing has been cancelled

# RVE (global)
rve_threshold = 0.25
rve_cmp = cv.imread(resource_path('seed/rve.png'))

# Load the player-side char images
# Note that these images are all oriented wrt P2 SIDE    
dir = 'uni_char/'
train_dir = dir + 'training/'
seed_dir = dir + 'seed/'
names = ['Akatsuki', 'Byakuya', 'Carmine', 'Chaos', 'Eltnum', 'Enkidu', 'Gordeau', 'Hilda', 'Hyde', 'Linne', 'Londrekia', 'Merkava', 'Mika', 'Nanase', 'Orie', 'Phonon', 'Seth', 'Vatista', 'Wagner', 'Waldstein', 'Yuzuriha']
ext = '.png'
# list of "char_imgs" dictionaries; each list element is a dictionary containing each character's CHAR_X.png, where X is on [1, 20]
locd = [ { name : cv.imread(resource_path(train_dir+name.lower()+'_'+str(num)+ext)) for name in names } for num in range(1, 21) ]
locd = np.array(locd)
#char_imgs = { name : cv.imread(resource_path(seed_dir+name+ext)) for name in names }
#locd.append(char_imgs)


# GUI
root = tkinter.Tk()
root.title('KAT GUI')
root.geometry('500x500')
root.resizable(tkinter.FALSE, tkinter.FALSE)
root.update()

# File name/URL Entry box
labelText = tkinter.StringVar()
label = tkinter.Entry(root, textvariable=labelText, justify='center')

# Dropdown Text -- YouTube URL or Local File
buttonText = tkinter.StringVar(root)
buttonText.set("YouTube URL")

# Progress
progressText = tkinter.StringVar(root)
progressText.set("x/x frames (- %)")
progress=tkinter.Label(root, textvariable=progressText)

# Instructions
instructionsTextYT = "Click the '[Paste link]' button to paste a YouTube URL from the clipboard. You can also type in a YouTube URL into the text entry box manually. Make sure the link includes 'http://' or 'https://'."
instructionsTextLcl = "Click the 'Browse' button to open a video file stored on disk, using your operating system's file dialog. You can also type in the path to the file manually."
instructions = tkinter.Label(root, text=instructionsTextYT, wraplength=root.winfo_width())

# Debug output window
debugOutput = tkinter.scrolledtext.ScrolledText(root, undo=True)
debugOutput['font'] = ('consolas', '12')

button = None

start = None

# Custom output stream for debug output window
class PrintLogger():
    def __init__(self, textbox):
        self.textbox = textbox
    
    def write(self, text):
        self.textbox.insert(tkinter.END, text)
        self.textbox.see("end")
        root.update_idletasks()
    
    def flush(self):
        pass

# Before closing window    
def on_exit():
    global execution
    if thrd != None:
        execution = False
        thrd.join()
    root.quit()
    root.destroy()

# Auto-timestamping function        
def divide(cap_, vidstr_, ytmode, safe_vidstr_):
    
    global execution
    global cancel
    
    fps = cap_.get(cv.CAP_PROP_FPS)
    if debug:
        print("FPS: {}".format(fps))
        
    frame_count = 0
    frame_skip = round(fps) / 5
    if debug:
        print("Frame skip: {}".format(frame_skip))
    
    total_frames = cap_.get(cv.CAP_PROP_FRAME_COUNT)
    
    width = int(cap_.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap_.get(cv.CAP_PROP_FRAME_HEIGHT))
    if debug:
        print("Resolution: {}x{}".format(width, height))
    
    # rectangle dims
    p1_midpoint_h = 0.0576171875 * width
    p1_midpoint_v = 0.0677083333333333 * height
    p1_left_h = int(p1_midpoint_h - 0.0400390625 * width)# + 12
    p1_right_h = int(p1_midpoint_h + 0.0400390625 * width)# + 12
    p1_left_v = int(p1_midpoint_v - 0.0677083333333333 * height)# + 68
    p1_right_v = int(p1_midpoint_v + 0.0677083333333333 * height)# + 68
        
    p2_midpoint_h = 0.9404296875 * width
    p2_midpoint_v = 0.0677083333333333 * height
    p2_left_h = int(p2_midpoint_h - 0.0380859375 * width)# + 12
    p2_right_h = int(p2_midpoint_h + 0.0380859375 * width)# + 12
    p2_left_v = int(p2_midpoint_v - 0.0677083333333333 * height)# + 68
    p2_right_v = int(p2_midpoint_v + 0.0677083333333333 * height)# + 68
    
    rve_midpoint_h = 0.5087890625 * width
    rve_midpoint_v = 0.4965277777777778 * height
    rve_left_h = int(rve_midpoint_h - 0.3251953125 * width)# + 12
    rve_right_h = int(rve_midpoint_h + 0.3251953125 * width)# + 12
    rve_left_v = int(rve_midpoint_v - 0.0416666666666667 * height)# + 68
    rve_right_v = int(rve_midpoint_v + 0.0416666666666667 * height)# + 68

    clause_left_h = int(0.490234375 * width)# + 12
    clause_left_v = int(0.1319444444444444 * height)# + 68
    clause_right_h = int(0.5390625 * width)# + 12
    clause_right_v = int(0.15625 * height)# + 68
        
    cnt = 0                     # to prevent checking too often
    rve_crop = None             # current crop of rve rectangle
    
    rve = False                     # determine when to take timestamp
    after_rve = False       # determine when to check character icons
    
    # Timestamps
    timestamps = []
    vs = []
    fname = ''
    if ytmode:
        fname = os.path.splitext(safe_vidstr_)[0] + '.txt'
#        fname = vidstr_.replace('/', ' ').replace(':', ' ')+'.txt'
    else:
        fname = os.path.basename(vidstr_+'.txt')
    f = open(resource_path('../' + fname), 'w', encoding='utf-8')
    print(vidstr_, file=f)
    print("TIMESTAMPS", file=f)
    
    # naive
    print("{} - Start".format(timedelta(seconds=0)), file=f)
    
    # Begin!
    print("Analysing...\n")
        
    while cap_.isOpened():
    
        # multithreading
        if not execution:
            cap_.release()
            cv.destroyAllWindows()
            return
        
        # cancel
        if cancel:
            break
    
        '''
            Keep track of which frame we are on.
            This must properly account for frame skip.
            These are used to keep track of timestamps.
        '''
        time = float(frame_count)/fps

        if cnt <= 500:
            cnt+=1
        else:
            cnt=1
        ret, frame = cap_.read()
    
        # ret is True if frame read correctly
        if not ret:
            if debug:
                print("Can't receive frame (end of stream?). Exiting...")
            else:
                print("Done.")
            break
    
        cv.rectangle(frame, (p1_left_h, p1_left_v), (p1_right_h, p1_right_v), (255, 0, 0), 2)       # P1 (blue)
        cv.rectangle(frame, (p2_left_h, p2_left_v), (p2_right_h, p2_right_v), (0, 0, 255), 2)       # P2 (red)
        cv.rectangle(frame, (rve_left_h, rve_left_v), (rve_right_h, rve_right_v), (0, 255, 0), 2)   # Recurring VOID Effect...
        cv.rectangle(frame, (clause_left_h, clause_left_v), (clause_right_h, clause_right_v), (0, 255, 0), 1) # CLAUSE
        
        if cnt % 5 == 0:
            rve_crop = frame[rve_left_v:rve_right_v, rve_left_h:rve_right_h, :]
            rve_crop = cv.resize(rve_crop, dsize=(rve_cmp.shape[1], rve_cmp.shape[0]), interpolation=cv.INTER_CUBIC)
                    
            (score, _) = compare_ssim(rve_crop, rve_cmp, full=True, multichannel=True)
                    
            if score >= 0.2 and not rve:
                if debug:
                    print("[potential miss?]")
                    print("SSIM: {}".format(score))
                    print("Timestamp: {}\n".format(timedelta(seconds=time)))
            if rve:
                if debug:
                    print("SSIM: {}".format(score))
                    print("Timestamp: {}\n".format(timedelta(seconds=time)))
            elif not rve and after_rve:
                # crop char icons; note that the 2 is added or subtracted to eliminate the blue/red border (rectangle thickness)
                p1_crop = frame[p1_left_v+2:p1_right_v-2, p1_left_h+2:p1_right_h-2, :]
                p2_crop = frame[p2_left_v+2:p2_right_v-2, p2_left_h+2:p2_right_h-2, :]
                
                # resize and flip horizontally the P1 crop
                p1_crop = cv.resize(p1_crop, dsize=(48,48), interpolation=cv.INTER_AREA)
                p1_crop = cv.flip(p1_crop, 1)
                
                # resize the P2 crop
                p2_crop = cv.resize(p2_crop, dsize=(48,48), interpolation=cv.INTER_AREA)
                
#                p1_crop = np.expand_dims(p1_crop, axis=0)
#                p2_crop = np.expand_dims(p2_crop, axis=0)
#                p1_crop = p1_crop.astype(np.float32)
#                p2_crop = p2_crop.astype(np.float32)
#                p1_crop /= 255.0
#                p2_crop /= 255.0

                best_score1 = -1.0
                best_score2 = -1.0
                best_name1 = ''
                best_name2 = ''
                for name in names:
                    p1_score = 0
                    p2_score = 0
                    for imgnum in range(20):
                        (tmp1_score, _) = compare_ssim(p1_crop, locd[imgnum][name], full=True, multichannel=True)
                        (tmp2_score, _) = compare_ssim(p2_crop, locd[imgnum][name], full=True, multichannel=True)
                        p1_score += tmp1_score
                        p2_score += tmp2_score
                    p1_score /= 20
                    p2_score /= 20
                
#                   (p1_score, _) = compare_ssim(p1_crop, char_imgs[name], full=True, multichannel=True)
#                   (p2_score, _) = compare_ssim(p2_crop, char_imgs[name], full=True, multichannel=True)

#                    cur_char = np.expand_dims(char_imgs[name], axis=0)
#                    p1_score = siamese_nn.predict([p1_crop, cur_char])[0][0]
#                    p2_score = siamese_nn.predict([p2_crop, cur_char])[0][0]
                    if p1_score > best_score1:
                        best_score1 = p1_score
                        best_name1 = name
                    if p2_score > best_score2:
                        best_score2 = p2_score
                        best_name2 = name
                
                if debug:
                    print("P1 - {} vs {} - P2".format(best_name1, best_name2))
                    print("P1 SSIM: {}".format(best_score1))
                    print("P2 SSIM: {}\n".format(best_score2))
                
                print("{} vs {}".format(best_name1, best_name2), file=f)
                vs.append(best_name1 + ' vs ' + best_name2)
                
                after_rve = False
                
            if score >= rve_threshold and not rve:
                if debug:
                    print("Divide!")
                    print("SSIM: {}".format(score))
                    print("Timestamp: {}\n".format(timedelta(seconds=time)))
                
                print("{} - ".format(timedelta(seconds=round(time))), file=f, end='')
                timestamps.append(time)
                
                rve = True
            
            if score < rve_threshold and rve:
                rve = False
                after_rve = True

        # advance by |frame_skip| frames
        for i in range(0, int(frame_skip)):
            cap_.grab()
        frame_count += frame_skip + 1

        progressText.set(str(frame_count)+"/"+str(total_frames)+" frames (" + str(round(frame_count/total_frames * 100)) + " %)")
#        if debug:
#            cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
            
    print("\nTIMESTAMPS")
    print("{} - Start".format(timedelta(seconds=0)))
    for i in range(0, len(timestamps)):
        print("{} - {}".format(timedelta(seconds=round(timestamps[i])), vs[i]))

    print("\nTimestamps written to '" + fname + "'")   
    f.close()
    
    # Cleanup
    cap_.release()
    cv.destroyAllWindows()
    execution = False
    
    # reenable starting a new timestamping session
    start.config(command=preinit)

# OS file dialog to obtain video file from on disk
def open_file():
    file_name = tkinter.filedialog.askopenfilename(parent=root, title='Choose a video file')
    labelText.set(file_name)

# Paste YT URL Link from clipboard    
def paste():
    labelText.set(root.clipboard_get())

# Change button functionality based on dropdown menu selection
def ddm_check(selection):
    if selection == 'YouTube URL':
        button.config(text='[Paste link]', command=paste)
        instructions.config(text=instructionsTextYT)
    elif selection == 'Local File':
        button.config(text='Browse', command=open_file)
        instructions.config(text=instructionsTextLcl)

# Execute BEFORE calling divide()
def preinit():
    # disable starting a new timestamping session
    start.config(command=tkinter.NONE)
    
    vidstr = ''
    cap = None
    
    global thrd
    global execution
    global cancel
    
    global pack
    
    ytmode = True # YouTube URL
    safe_vidstr = ''
    
    if not pack:
        debugOutput.pack(expand=True, fill='both', side=tkinter.BOTTOM)
        pl = PrintLogger(debugOutput)
        sys.stdout = pl
        pack = True
    else:
        debugOutput.delete("1.0", tkinter.END)
        
    vidstr = labelText.get()
    
    if "https://" in vidstr or "http://" in vidstr:
#        urlPafy = pafy.new(vidstr)
#        videoplay = urlPafy.getbest()
#        cap = cv.VideoCapture(videoplay.url)
#        vidstr = urlPafy.title
        video = pytube.YouTube(vidstr)
        stream = video.streams.get_highest_resolution()
        cap = cv.VideoCapture(stream.url)
        vidstr = video.title
        safe_vidstr = stream.default_filename
        ytmode = True
    else:
        cap = cv.VideoCapture(vidstr)
        ytmode = False
        
    if cap.isOpened():
        cancel = False
        thrd = threading.Thread(target=divide, args=(cap, vidstr, ytmode, safe_vidstr))
        thrd.start()
        execution = True
    else:
        print("Error opening video capture.")

# cancel running execution
def fcancel():
    global cancel
    cancel = True

ddm = tkinter.OptionMenu(root, buttonText, 'YouTube URL', 'Local File', command=ddm_check)    
button = tkinter.Button(root, text="[Paste link]", command=paste)
start = tkinter.Button(root, text="Start", command=preinit)
cancelButton = tkinter.Button(root, text="Cancel", command=fcancel)

ddm.pack()
instructions.pack()
button.pack()
label.pack(ipadx=root.winfo_width()/4)
cancelButton.pack(side=tkinter.BOTTOM)
start.pack(side=tkinter.BOTTOM)
progress.pack()

root.protocol("WM_DELETE_WINDOW", on_exit)
root.mainloop()
