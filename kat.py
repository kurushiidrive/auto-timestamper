#
# Kurushii Drive's Auto-Timestamper for UNI (KAT-UNI)
# 18 Aug 2021
# 
# Notes:
# - having problems classifying Yuzu and Orie.
# - Yuzu sometimes gets classified as Carmine
# - Orie sometimes gets classified as Seth
# - these misclassifications are more likely on P2 side
# - P2 side in general has lower best similarity than P1

import cv2 as cv
import numpy as np
import sys
#import pytesseract
from datetime import timedelta
from skimage.metrics import structural_similarity as compare_ssim
import pafy
import youtube_dl

debug = True
p1_writes = 0
p2_writes = 0

#pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\kurus\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

#vidstr = '旧UNI 2_10秋葉原対戦会 part1-sm20071603.mp4'
#vidstr = '旧UNI 2_10秋葉原対戦会 part2-sm20071786.mp4'
#vidstr = '旧UNI 2_10秋葉原対戦会 part3-sm20072152.mp4'
#vidstr = '旧UNI 2_10秋葉原対戦会 part4-sm20107916.mp4'
#vidstr = '旧UNI 2_10秋葉原対戦会 part5-sm20108259.mp4'
#vidstr = '旧UNI 2_10秋葉原対戦会 part6-sm20108535.mp4'

#vidstr = 'a-cho アンダーナイトーンヴァース　ランダム2on2大会（2013.8.4）-OyE1KCbbfJY.mp4'
#vidstr = 'a-cho　アンダーナイトインヴァース　ランダム2on2大会（2013.7.15）-wYkAYKzCBl0.mp4'
#vidstr = 'a-cho UNDER NIGHT IN-BIRTH 『BARRACUDA』 a-cho予選（2012.11.4）-9F5RjtOir-k.mkv'
#vidstr = 'a-cho　アンダーナイトインヴァース　ランダム2on2大会（2013.6.9）-rgJfuPssTyo.mp4'
#vidstr = '10月6日 アテナ日本橋 UNI大会 予選リーグB-sm19062518.mp4'
#vidstr = '10月6日 アテナ日本橋 UNI大会 予選リーグC-sm19062804.mp4'
#vidstr = '10月6日 アテナ日本橋 UNI大会 予選リーグD-sm19062663.mp4'
#vidstr = '12月21日 アテナ日本橋 UNIランバト 決勝トーナメント1_2-sm19648317.mp4'
#vidstr = '12月21日 アテナ日本橋 UNIランバト 決勝トーナメント2_2-sm19648429.mp4'
#vidstr = '旧UNI 2_16蒲田対戦会 part1-sm20109087.mp4'
#vidstr = '旧UNI 2_16蒲田対戦会 part2-sm20109217.mp4'

if len(sys.argv) != 2:
    print("Usage:\tpython {} [video]".format(sys.argv[0]))
    exit()

vidstr = sys.argv[1]
cap = None

if "https://" in vidstr:
    urlPafy = pafy.new(vidstr)
    videoplay = urlPafy.getbest()
    cap = cv.VideoCapture(videoplay.url)
    vidstr = urlPafy.title
else:
    cap = cv.VideoCapture(vidstr)

cnt = 0
crop = None

fps = cap.get(cv.CAP_PROP_FPS)
if debug:
    print("FPS: {}".format(fps))
frame_count = 0

rve = False                     # determine when to take timestamp
after_rve = False       # determine when to check character icons
rve_threshold = 0.5

cmp_image = cv.imread('rve.png')

frame_skip = round(fps) / 5
skip = 0
if debug:
    print("Frame skip: {}".format(frame_skip))

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
if debug:
    print("Resolution: {}x{}".format(width, height))

# UNI (vanilla) testing -- values calibrated @ 512x288
# rve horizontal midpoint @ 0.5087890625 * width
# rve horizontal half-dist: 0.3251953125 * width
# rve vertical midpoint @ 0.4965277777777778 * height
# rve vertical half-dist: 0.0416666666666667 * height
# P1 horizontal midpoint @ 0.0576171875 * width
# P1 horizontal half-dist: 0.0400390625 * width
# P1 vertical midpoint @ 0.0677083333333333 * height
# P1 vertical half-dist: 0.0677083333333333 * height
# P2 horizontal midpoint @ 0.9404296875 * width
# P2 horizontal half-dist: 0.0380859375 * width
# P2 vertical midpoint @ 0.0677083333333333 * height
# P2 vertical half-dist: 0.0677083333333333 * height
# CLAUSE left_h: 0.490234375 * width
# CLAUSE left_v: 0.1319444444444444 * height
# CLAUSE right_h: 0.5390625 * width
# CLAUSE right_v: 0.15625 * height
   
p1_midpoint_h = 0.0576171875 * width
p1_midpoint_v = 0.0677083333333333 * height
p1_left_h = int(p1_midpoint_h - 0.0400390625 * width)
p1_right_h = int(p1_midpoint_h + 0.0400390625 * width)
p1_left_v = int(p1_midpoint_v - 0.0677083333333333 * height)
p1_right_v = int(p1_midpoint_v + 0.0677083333333333 * height)
    
p2_midpoint_h = 0.9404296875 * width
p2_midpoint_v = 0.0677083333333333 * height
p2_left_h = int(p2_midpoint_h - 0.0380859375 * width)
p2_right_h = int(p2_midpoint_h + 0.0380859375 * width)
p2_left_v = int(p2_midpoint_v - 0.0677083333333333 * height)
p2_right_v = int(p2_midpoint_v + 0.0677083333333333 * height)
   
rve_midpoint_h = 0.5087890625 * width
rve_midpoint_v = 0.4965277777777778 * height
rve_left_h = int(rve_midpoint_h - 0.3251953125 * width)
rve_right_h = int(rve_midpoint_h + 0.3251953125 * width)
rve_left_v = int(rve_midpoint_v - 0.0416666666666667 * height)
rve_right_v = int(rve_midpoint_v + 0.0416666666666667 * height)

clause_left_h = int(0.490234375 * width)
clause_left_v = int(0.1319444444444444 * height)
clause_right_h = int(0.5390625 * width)
clause_right_v = int(0.15625 * height)

print()

# Load the player-side char images
# Note that these images are all oriented wrt P2 SIDE
dir = 'uni_char/'
train_dir = dir + 'training/'
names = ['Akatsuki', 'Carmine', 'Chaos', 'Eltnum', 'Gordeau', 'Hilda', 'Hyde', 'Linne', 'Merkava', 'Orie', 'Seth', 'Vatista', 'Waldstein', 'Yuzuriha']
ext = '.png'
char_imgs = { name : cv.cvtColor(cv.imread(dir+name+ext), cv.COLOR_BGR2GRAY) for name in names }

# Timestamps
timestamps = []
vs = []
f = open(vidstr+'.txt', 'w')

print(vidstr, file=f)
print("TIMESTAMPS", file=f)

# naive
print("{} - Start".format(timedelta(seconds=0)), file=f)

print("Analysing...\n")

while cap.isOpened():

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
    ret, frame = cap.read()
    
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

    # old hardcoded values @ 512x288
#    cv.rectangle(frame, (9, 0), (50, 39), (255, 0, 0), 2)                                       # P1 (blue)
#    cv.rectangle(frame, (frame.shape[1]-50, 0), (frame.shape[1]-11, 39), (0, 0, 255), 2)        # P2 (red)    
#    cv.rectangle(frame, (94, 131), (427, 155), (0, 255, 0), 2)                              # Recurring VOID Effect...
    
    
#    if cnt == 350:
#        cv.imwrite('tmpuni_360.png', frame)
#    if cnt == 350:
#        cv.imwrite('CLAUSE.png', frame[clause_left_v+1:clause_right_v-1, clause_left_h+1:clause_right_h-1, :])
    
    # for UNI, based on light testing, a similarity of 0.5 or above feels good.
    # the first time it hits 0.5, take a timestamp and set a flag.
    # once it goes below 0.5, unset the flag.
    # for more details, see the state machine visual on github
    #
    # NOTE:
    # Structural Similarity is the approach we'll take for now.
    # might test siamese networks later.
    #
    # Check and process once every second.
    if cnt % 5 == 0:
        crop = frame[rve_left_v:rve_right_v, rve_left_h:rve_right_h, :]
        crop = cv.resize(crop, dsize=(cmp_image.shape[1], cmp_image.shape[0]), interpolation=cv.INTER_CUBIC)
        (score, diff) = compare_ssim(crop, cmp_image, full=True, multichannel=True)
        diff = (diff * 255).astype("uint8")
        if score >= 0.4 and not rve:
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
            
#            cv.imwrite(train_dir + 'p1_' + str(p1_writes) + '.png', p1_crop)
#            cv.imwrite(train_dir + 'p2_' + str(p2_writes) + '.png', p2_crop)
            p1_writes += 1
            p2_writes += 1
            
            p1_crop = cv.cvtColor(p1_crop, cv.COLOR_BGR2GRAY)
            p2_crop = cv.cvtColor(p2_crop, cv.COLOR_BGR2GRAY)
            
            best_score1 = -1.0
            best_score2 = -1.0
            best_name1 = ''
            best_name2 = ''
            for name in names:
                (p1_score, _) = compare_ssim(p1_crop, char_imgs[name], full=True, multichannel=True)
                (p2_score, _) = compare_ssim(p2_crop, char_imgs[name], full=True, multichannel=True)
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
        cap.grab()
    frame_count += frame_skip + 1
        
    # cap.set is less stable, and has more performance overhead for frame advancing
#    cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break


print("\nTIMESTAMPS")
print("{} - Start".format(timedelta(seconds=0)))
for i in range(0, len(timestamps)):
    print("{} - {}".format(timedelta(seconds=round(timestamps[i])), vs[i]))
   
print("\nTimestamps written to '" + vidstr + ".txt'")   
f.close()

cap.release()
cv.destroyAllWindows()