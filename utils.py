import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def make_pairs(images, labels):
    # create two empty lists to hold the (img, img) pairs
    # labels indicate if pair is positive (imgs are the same) or negative (imgs are different)
    pair_images = []
    pair_labels = []
    
    # calculate number of classes in the dataset
    num_classes = len(np.unique(labels))
    
    # build a list of indices for each class label to provide the indices for all examples with a given label
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]
    
    for idx_a in range(len(images)):
        cur_img = images[idx_a]
        label = labels[idx_a]
        
        idx_b = np.random.choice(idx[label])
        pos_img = images[idx_b]
        
        pair_images.append([cur_img, pos_img])
        pair_labels.append([1])
        
        neg_idx = np.where(labels != label)[0]
        neg_img = images[np.random.choice(neg_idx)]
        
        pair_images.append([cur_img, neg_img])
        pair_labels.append([0])
    
    return (np.array(pair_images), np.array(pair_labels))
    
def euclidean_distance(vectors):
    (features_a, features_b) = vectors
    
    squared_sum = K.sum(K.square(features_a - features_b), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(squared_sum, K.epsilon()))
    
def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss
    
def plot_training(H, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history['loss'], label='train_loss')
#    plt.plot(H.history['val_loss'], label='val_loss')
    plt.plot(H.history['acc'], label='train_acc')
#    plt.plot(H.history['val_acc'], label='val_acc')
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)