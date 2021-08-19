from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

def build_siamese_model(inputShape, embeddingDim=48):
    inputs = Input(inputShape)
    
    # first set of Conv -> Relu -> Pool -> Dropout layers
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    
    # second set of C -> R -> P -> D layers
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    
    model = Model(inputs, outputs)
    
    return model