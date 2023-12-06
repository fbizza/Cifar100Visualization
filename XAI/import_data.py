import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from model import compute_features_vectors, predict_fine_class
coarse_to_category = {
    0: 'aquatic mammals',
    1: 'fish',
    2: 'flowers',
    3: 'food containers',
    4: 'fruit and vegetables',
    5: 'household electrical devices',
    6: 'household furniture',
    7: 'insects',
    8: 'large carnivores',
    9: 'large man-made outdoor things',
    10: 'large natural outdoor scenes',
    11: 'large omnivores and herbivores',
    12: 'medium-sized mammals',
    13: 'non-insect invertebrates',
    14: 'people',
    15: 'reptiles',
    16: 'small mammals',
    17: 'trees',
    18: 'vehicles 1',
    19: 'vehicles 2'
}
fine_to_cateogry = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'computer_keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}

np.random.seed(seed=666)

(x_train_fine, fine_labels_train), (x_test_fine, fine_labels_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(x_train_coarse, coarse_labels_train), (x_test_coarse, coarse_labels_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

def tsne_data(N_IMAGES_PER_CLASS):

    # Loop over each class label and sample N_IMAGES_PER_CLASS random images over each class
    idx = np.empty(0, dtype="int8")
    for i in range(0, len(np.unique(coarse_labels_test))):
        idx = np.append(idx, np.random.choice(np.where((coarse_labels_test[0:len(coarse_labels_test)]) == i)[0],
                                              N_IMAGES_PER_CLASS, replace=False))

    x_test = x_test_coarse[idx]
    y_test_coarse = coarse_labels_test[idx]
    y_test_fine = fine_labels_test[idx]

    # #### ----- Uncomment this part to plot an example image for each class ----- #####
    #
    # plt.figure(figsize=(20, 20))
    # for i in range(len(np.unique(y_train))):
    #     class_indices = np.where(y_train == i)[0]
    #     random_index = np.random.choice(class_indices, 1)
    #     plt.subplot(1, 20, i + 1)
    #     img = x_train[random_index].reshape(32, 32, 3)
    #     plt.imshow(img)
    #     plt.axis('off')
    # plt.show()
    print("Predicting classes...")
    predicted_fine_categories = predict_fine_class(x_test)
    print("Extracting feature vectors...")
    first_block_tsne_x, first_block_tsne_y = tsne_intermediate_layer(x_test, 'max_pooling2d')
    second_block_tsne_x, second_block_tsne_y = tsne_intermediate_layer(x_test, 'max_pooling2d_2')
    third_block_tsne_x, third_block_tsne_y = tsne_intermediate_layer(x_test, 'max_pooling2d_3')
    fourth_block_tsne_x, fourth_block_tsne_y = tsne_intermediate_layer(x_test, 'max_pooling2d_4')
    softmax_tsne_x, softmax_tsne_y = tsne_intermediate_layer(x_test, 'activation_14')
    coarse_labels = y_test_coarse  # These are numbers from 0 to 19
    fine_labels = y_test_fine  # These are numbers from 0 to 99
    coarse_categories = [coarse_to_category[label] for label in coarse_labels.flatten()]  # These are strings
    fine_categories = [fine_to_cateogry[label] for label in fine_labels.flatten()]  # These are strings
    

    return (softmax_tsne_x, softmax_tsne_y,                                         # t-sne of feature vector
            first_block_tsne_x, first_block_tsne_y,                                 # t-sne of feature vector
            second_block_tsne_x, second_block_tsne_y,                               # t-sne of feature vector
            third_block_tsne_x, third_block_tsne_y,                                 # t-sne of feature vector
            fourth_block_tsne_x, fourth_block_tsne_y,                               # t-sne of feature vector
            predicted_fine_categories,                                              # Predicted labels
            coarse_labels.flatten().tolist(), coarse_categories, fine_categories,   # Ground truth labels
            x_test.reshape((len(x_test), 32 * 32 * 3)))                           # Raw image pixels


def tsne_intermediate_layer(x_train, layer_name):
    images = compute_features_vectors(x_train, layer_name)
    model = TSNE(n_components=2)
    tsne = model.fit_transform(images.reshape((len(x_train), -1)))
    x = tsne[:, 0]
    y = tsne[:, 1]
    return x, y




