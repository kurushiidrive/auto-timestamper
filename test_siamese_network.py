import cv2 as cv
import numpy as np
import utils
import config
from tensorflow import keras
from imutils import build_montages
from siamese_network import build_siamese_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

names = ['akatsuki', 'byakuya', 'carmine', 'chaos', 'eltnum', 'enkidu', 'gordeau', 'hilda', 'hyde', 'linne', 'londrekia', 'merkava', 'mika', 'nanase', 'orie', 'phonon', 'seth', 'vatista', 'wagner', 'waldstein', 'yuzuriha']
nametocode = { names[i] : i for i in range(len(names)) }

dir = 'uni_char/test/'
tmpX = []
tmpY = []
for name in names:
    for count in range(1,3):
        tmpX.append(cv.imread(dir+name+'_'+str(count)+'.png'))
        tmpY.append(nametocode[name])
tmpX = np.array(tmpX)
X_test = tmpX / 255.0
y_test = np.array(tmpY)

(pair_test, label_test) = utils.make_pairs(X_test, y_test)
(tmp_pair_test, tmp_label_test) = utils.make_pairs(tmpX, y_test)

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

#model = keras.models.load_model(config.MODEL_NAME)

x_test_1 = pair_test[:, 0]
x_test_2 = pair_test[:, 1]

results = model.evaluate([x_test_1, x_test_2], label_test)
print("test loss, test acc:", results)

predictions = model.predict([x_test_1, x_test_2])
print(predictions)
print(predictions.shape)

images = []

for i in range(predictions.shape[0]):
    imageA = tmp_pair_test[i][0]
    imageB = tmp_pair_test[i][1]
#    label = tmp_label_train[i]
    
    # padding for easier visualization
#    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([imageA, imageB])
#    output[4:32, 0:56] = pair
    
    text = "neg" if predictions[i][0] < 0.5 else "pos"
    color = (0, 0, 255) if predictions[i][0] < 0.5 else (0, 255, 0)
    
#    vis = cv.merge([output] * 3)    # 3-channel RGB image from original grayscale
    vis = cv.resize(pair, (96, 51), interpolation=cv.INTER_LINEAR)
    cv.putText(vis, text, (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    images.append(vis)
    
montage = build_montages(images, (96, 51), (7, 7))[0]

cv.imshow("Siamese Image Pairs", montage)
while cv.waitKey(0) != ord('q'):
    pass