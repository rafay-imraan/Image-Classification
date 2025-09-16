# IMAGE CLASSIFICATION USING LOGISTIC REGRESSION

# Import libraries
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Declaring constants
IMAGE_SIZE = (150, 150)
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Label mapping
label_map = {name: idx for idx, name in enumerate(CLASSES)}
inv_label_map = {v: k for k, v in label_map.items()}

# Load images
def load_dataset(folder):
    data = [] 
    labels = []
    for class_name in CLASSES:
        class_folder = os.path.join(folder, class_name)
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(".jpg"):
                img_path = os.path.join(class_folder, fname)
                img = Image.open(img_path).resize(IMAGE_SIZE).convert("L")
                img_array = np.array(img).flatten()
                data.append(img_array)
                labels.append(label_map[class_name])
    return np.array(data), np.array(labels)

# Load training and testing sets
X_train, y_train = load_dataset("seg_train")
X_test, y_test = load_dataset("seg_test")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality
pca = PCA(n_components = 500)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.fit_transform(X_test_scaled)

# Create and train the model
model = LogisticRegression(max_iter = 2000)
model.fit(X_train_pca, y_train)

# Evaluate
y_pred = model.predict(X_test_pca)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names = CLASSES))