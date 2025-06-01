# import numpy as np
# import tensorflow as tf
# import cv2

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
#     # Create a model that maps the input image to the activations
#     # of the last conv layer and the output predictions
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, 0]

#     # Compute gradients of the top predicted class with regard to
#     # the output feature map of the last conv layer
#     grads = tape.gradient(loss, conv_outputs)

#     # Mean intensity of the gradients over the spatial dimensions
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]

#     # Weight the feature maps by the importance of the gradients
#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#     # Relu on heatmap (only positive values)
#     heatmap = tf.maximum(heatmap, 0)

#     # Normalize heatmap to range 0-1
#     heatmap /= tf.math.reduce_max(heatmap) + 1e-8  # add epsilon to avoid division by zero

#     return heatmap.numpy()

# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#     # Read original image
#     img = cv2.imread(img_path)

#     # Resize to match model input size (32x32)
#     img = cv2.resize(img, (32, 32))

#     # Resize heatmap to match image size
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#     # Convert heatmap to RGB (apply colormap)
#     heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     # Overlay heatmap on image
#     overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

#     # Save the image with heatmap overlay
#     cv2.imwrite(cam_path, overlayed_img)

#     return cam_path
