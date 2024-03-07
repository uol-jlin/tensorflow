import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

source_path = '/tmp/PetImages'

source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')

# Deletes all non-image files (there are two .db files bundled into the dataset)
!find /tmp/PetImages/ -type f ! -name "*.jpg" -exec rm {} +

# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")
    
# Define root directory
root_dir = '/tmp/cats-v-dogs'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

def create_train_val_dirs(root_path):
  """
  Creates directories for the train and test sets
  
  Args:
    root_path (string) - the base directory path to create subdirectories from
  
  Returns:
    None
  """

  training_dir = os.path.join(root_path, "training")
  validation_dir = os.path.join(root_path, "validation")

  os.makedirs(training_dir)
  os.makedirs(validation_dir)

  for dir in [training_dir, validation_dir]:
    os.makedirs(os.path.join(dir, "cats"))
    os.makedirs(os.path.join(dir, "dogs"))

try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets
  
  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training
    
  Returns:
    None
  """
  source_files = [file for file in os.listdir(SOURCE_DIR) if os.path.getsize(os.path.join(SOURCE_DIR, file)) > 0]
  source_files = random.sample(source_files, len(source_files))

  split_index = int(len(source_files) * SPLIT_SIZE)
  training_files = source_files[:split_index]
  validation_files = source_files[split_index:]
  
  for file in training_files:
    copyfile(os.path.join(SOURCE_DIR, file), os.path.join(TRAINING_DIR, file))
    
  for file in validation_files:
    copyfile(os.path.join(SOURCE_DIR, file), os.path.join(VALIDATION_DIR, file))
