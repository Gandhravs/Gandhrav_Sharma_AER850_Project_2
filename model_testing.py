from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'model1.h5'  # Path to your saved model file
CNN_model = load_model(model_path)

# Test images directory path
test_image_paths = [
    r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\test\crack\IMG_20230511_101043_jpg.rf.0f754b4a1df6afcfed04bb8468a5f2cb.jpg",
    r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\test\missing-head\IMG_20230511_100229_jpg.rf.08e4a8127f1d2057801e4a7087862f85.jpg",
    r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\test\paint-off\IMG_20230511_095534_jpg.rf.3fc59d9b5d2995245d842b30d302e095.jpg"
]

class_names = ['crack', 'missing-head', 'paint-off']  # Adjust class names as per your model's labels

for img_path in test_image_paths:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(100, 100))  # Use the target size used in training
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = CNN_model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Find the class with the highest probability
    confidence = predictions[0][predicted_class]

    # Display the result
    print(f"Image: {img_path}")
    print(f"Predicted class: {class_names[predicted_class]} with confidence {confidence:.2f}")

    # Optionally, display the image with the prediction
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
