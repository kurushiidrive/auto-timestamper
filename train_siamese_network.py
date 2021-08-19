from siamese_network import build_siamese_model
import config
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2 as cv
from imutils import build_montages

names = ['carmine', 'eltnum', 'gordeau', 'hilda', 'hyde', 'linne', 'merkava', 'orie', 'seth', 'vatista', 'waldstein', 'yuzuriha']
nametocode = { names[i] : i for i in range(12) }

dir = 'uni_char/training/'
tmpX = []
tmpY = []
for name in names:
    for count in range(1,4):
        tmpX.append(cv.imread(dir+name+'_'+str(count)+'.png'))
        tmpY.append(nametocode[name])
tmpX = np.array(tmpX)
X = tmpX / 255.0
y = np.array(tmpY)

(pairTrain, labelTrain) = utils.make_pairs(X, y)
(tmpPairTrain, tmpLabelTrain) = utils.make_pairs(tmpX, y)

'''
# load mnist and scale to [0,1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# for adding a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)
'''

print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation='sigmoid')(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

print("[INFO] compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print("[INFO] training model...")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
#    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS)
    
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)


# siamese pairs visualization
images = []

for i in np.random.choice(np.arange(0, len(tmpPairTrain)), size=(49,)):
    imageA = tmpPairTrain[i][0]
    imageB = tmpPairTrain[i][1]
    label = tmpLabelTrain[i]
    
    # padding for easier visualization
#    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([imageA, imageB])
#    output[4:32, 0:56] = pair
    
    text = "neg" if label[0] == 0 else "pos"
    color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
    
#    vis = cv.merge([output] * 3)    # 3-channel RGB image from original grayscale
    vis = cv.resize(pair, (96, 51), interpolation=cv.INTER_LINEAR)
    cv.putText(vis, text, (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    images.append(vis)
    
montage = build_montages(images, (96, 51), (7, 7))[0]

cv.imshow("Siamese Image Pairs", montage)
while cv.waitKey(0) != ord('q'):
    pass