import cv2 as cv
import numpy as np
import sys
from datetime import timedelta
import utils
import config
from tensorflow import keras
from siamese_network import build_siamese_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from skimage.metrics import structural_similarity as compare_ssim
import pafy
import youtube_dl

debug = True

# Setup
if len(sys.argv) != 2:
    print("Usage:\tpython {} [video]".format(sys.argv[0]))
    exit()

vidstr = sys.argv[1]
cap = None

if "https://" in vidstr or "http://" in vidstr:
    urlPafy = pafy.new(vidstr)
    videoplay = urlPafy.getbest()
    cap = cv.VideoCapture(videoplay.url)
    vidstr = urlPafy.title
else:
    cap = cv.VideoCapture(vidstr)

def reconstruct_model():
    # rebuild model
    imgA = Input(shape=config.IMG_SHAPE)
    imgB = Input(shape=config.IMG_SHAPE)
    featureExtractor = build_siamese_model(config.IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    distance = Lambda(utils.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation='sigmoid')(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss=utils.loss(margin=config.MARGIN), optimizer='RMSprop', metrics=['acc'])
    model.load_weights(config.MODEL_WEIGHTS_PATH)
    
    return model

# Model
siamese_nn = reconstruct_model()

# RVE (global)
rve_threshold = 0.4
rve_cmp = cv.imread('rve.png')

# Load the player-side char images
# Note that these images are all oriented wrt P2 SIDE    
dir = 'uni_char/'
names = ['Akatsuki', 'Byakuya', 'Carmine', 'Chaos', 'Eltnum', 'Enkidu', 'Gordeau', 'Hilda', 'Hyde', 'Linne', 'Londrekia', 'Merkava', 'Mika', 'Nanase', 'Orie', 'Phonon', 'Seth', 'Vatista', 'Wagner', 'Waldstein', 'Yuzuriha']
ext = '.png'
char_imgs = { name : cv.imread(dir+name+ext)/255.0 for name in names }

# Main loop; call each video's cap on this function to analyse them
def loop(cap_, vidstr_):
    fps = cap_.get(cv.CAP_PROP_FPS)
    if debug:
        print("FPS: {}".format(fps))
        
    frame_count = 0
    frame_skip = round(fps) / 5
    if debug:
        print("Frame skip: {}".format(frame_skip))
    
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
    f = open(vidstr_.replace('/', ' ')+'.txt', 'w', encoding='utf-8')
    print(vidstr_, file=f)
    print("TIMESTAMPS", file=f)
    
    # naive
    print("{} - Start".format(timedelta(seconds=0)), file=f)
    
    # Begin!
    print("Analysing...\n")
        
    while cap_.isOpened():
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
                
                p1_crop = np.expand_dims(p1_crop, axis=0)
                p2_crop = np.expand_dims(p2_crop, axis=0)
                p1_crop = p1_crop.astype(np.float32)
                p2_crop = p2_crop.astype(np.float32)
                p1_crop /= 255.0
                p2_crop /= 255.0

                best_score1 = -1.0
                best_score2 = -1.0
                best_name1 = ''
                best_name2 = ''
                for name in names:
#                    (p1_score, _) = compare_ssim(p1_crop, char_imgs[name], full=True, multichannel=True)
#                    (p2_score, _) = compare_ssim(p2_crop, char_imgs[name], full=True, multichannel=True)
                    cur_char = np.expand_dims(char_imgs[name], axis=0)
                    p1_score = siamese_nn.predict([p1_crop, cur_char])[0][0]
                    p2_score = siamese_nn.predict([p2_crop, cur_char])[0][0]
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

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
            
    print("\nTIMESTAMPS")
    print("{} - Start".format(timedelta(seconds=0)))
    for i in range(0, len(timestamps)):
        print("{} - {}".format(timedelta(seconds=round(timestamps[i])), vs[i]))

    print("\nTimestamps written to '" + vidstr_.replace('/', ' ') + ".txt'")   
    f.close()

loop(cap, vidstr)
cap.release()
cv.destroyAllWindows()