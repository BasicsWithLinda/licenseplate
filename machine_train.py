import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_otsu

# list of letters (labels)
letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def read_training_data(training_directory):
    """
    Reads training data from the given directory.
    Each letter is expected to have its own folder containing images.
    """
    image_data = []
    target_data = []
    
    # path to the train20X20 subdirectory
    train_dir = os.path.join(training_directory, 'train20X20')
    
    for each_letter in letters:
        letter_dir = os.path.join(train_dir, each_letter)
        if not os.path.isdir(letter_dir):
            continue  # skip if the directory doesn't exist
        for file_name in os.listdir(letter_dir):
            if file_name.endswith(('.jpg', '.png')):  # ensure supported formats
                image_path = os.path.join(letter_dir, file_name)
                img_details = imread(image_path, as_gray=True)
                binary_image = img_details < threshold_otsu(img_details)
                
                flat_bin_image = binary_image.reshape(-1)  # flatten the image
                image_data.append(flat_bin_image)
                target_data.append(each_letter)  # label is the letter (folder name)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    """
    Performs cross-validation and prints accuracy for each fold.
    """
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print(f"Cross-validation result for {num_of_fold}-fold:")
    print(accuracy_result * 100)

# directory paths
current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, 'training_data')

# loading the train data
image_data, target_data = read_training_data(training_dataset_dir)

# using svc model for supervised learning algorithm
svc_model = SVC(kernel='linear', probability=True)

# cross-validation using 4 fold method whereby 75% is used to train and 25% is to test the data and repeated 4 times 
cross_validation(svc_model, 4, image_data, target_data)

# train the model
svc_model.fit(image_data, target_data)

# save trained model
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, os.path.join(save_directory, 'svc.pkl'))

print("Model training complete. Saved to 'models/svc/svc.pkl'")
