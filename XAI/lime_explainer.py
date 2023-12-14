from model import predict_classes
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries


def predict(x):
    return predict_classes(x, norm=True)

# Creates LIME explanation image
def lime_explain_img(img):
    img = image.img_to_array(img)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img.astype('double'), predict, top_labels=3, hide_color=0, num_samples=100)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3,
                                                hide_rest=True)
    return mark_boundaries((temp).astype(np.uint8), (mask).astype(np.uint8))


# Use this function to test LIME without running the app
def test_lime_framework():
    img_path = "clicked-image.png"  # Change path to your 32x32 test image
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img.astype('double'), predict, top_labels=3, hide_color=0, num_samples=200)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3,
                                                hide_rest=True)

    plt.imshow(mark_boundaries((temp).astype(np.uint8), (mask).astype(np.uint8)))
    plt.axis('off')
    plt.show()
