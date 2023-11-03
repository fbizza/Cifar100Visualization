import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

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

np.random.seed(seed=666)

(x_train_fine, fine_labels_train), (x_test_fine, fine_labels_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(x_train_coarse, coarse_labels_train), (x_test_coarse, coarse_labels_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

def tsne_data(N_IMAGES_PER_CLASS=10):

    # Loop over each class label and sample N_IMAGES_PER_CLASS random images over each class
    idx = np.empty(0, dtype="int8")
    for i in range(0, len(np.unique(coarse_labels_train))):
        idx = np.append(idx, np.random.choice(np.where((coarse_labels_train[0:len(coarse_labels_train)]) == i)[0],
                                              N_IMAGES_PER_CLASS, replace=False))

    x_train = x_train_coarse[idx]
    y_train = coarse_labels_train[idx]

    ##### ----- Uncomment this part to plot an example image for each class ----- #####

    # plt.figure(figsize=(20, 20))
    # for i in range(len(np.unique(y_train))):
    #     class_indices = np.where(y_train == i)[0]
    #     random_index = np.random.choice(class_indices, 1)
    #     plt.subplot(1, 20, i + 1)
    #     img = x_train[random_index].reshape(32, 32, 3)
    #     plt.imshow(img)
    #     plt.axis('off')
    # plt.show()

    model = TSNE(n_components=2, random_state=0)
    tsne = model.fit_transform(x_train.reshape((len(x_train), 32 * 32 * 3)))  # Flatten the images
    x = tsne[:, 0]
    y = tsne[:, 1]
    coarse_labels = y_train  # These are numbers from 0 to 19
    coarse_categories = [coarse_to_category[label] for label in coarse_labels.flatten()]  # These are strings

    return x, y, coarse_labels.flatten().tolist(), coarse_categories, x_train.reshape((len(x_train), 32 * 32 * 3))




