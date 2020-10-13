# Data Exploration & Handling Challenges of Medical Image Data

In this resository we will discuss the following things step-by-step:
- [Data Exploration of **ChestX-ray8** dataset](https://nbviewer.jupyter.org/github/kampaitees/Medical-Data-Exploration-Handling-Challenges/blob/main/ChestX-ray8%20Data%20Exploration.ipynb)
- [Handling **Data Imbalance** within classes of a dataset](https://nbviewer.jupyter.org/github/kampaitees/Medical-Data-Exploration-Handling-Challenges/blob/main/Handling%20Data%20Imbalance%20using%20Weighted%20Loss%20Function.ipynb)
- [Creating pipeline for training **Densenet** on Medical Data](https://nbviewer.jupyter.org/github/kampaitees/Medical-Data-Exploration-Handling-Challenges/blob/main/Pipeline%20for%20training%20Densenet.ipynb)
- [**Handling Patient Overlap**(**Data Leakage**) problem in dataset](https://nbviewer.jupyter.org/github/kampaitees/Medical-Data-Exploration-Handling-Challenges/blob/main/Handling%20Patient%20Overlap%20%26%20Data%20Leakage.ipynb)
<br>

# Data Exploration of ChestX-ray8 dataset 

**While doing data exploration, we have to follow certain steps which I am going to discuss below:**

1) We have to load the dataset and have a look at it

2) We have to check what are different categories of data types are there in the dataset such as: categorical, continous, etc...

3) As we have seen the data, now we have to remove the useless data which won't help in exploration such as patient_id, image_url etc..

4) We alos have to check whehter there is duplcate entries are there in the dataset, if it's there then remove it.

5) Data Visualization - exploring the images it's histogram. It's important to see the distribution of image because most of the ML algo works best at 0 **Mean** and unit **Standard deviation**. To **Standardize** the dataset we can use **ImageDataGenerator** from **Keras**.

**We have seen the steps which are requrired to follow in data exploration of dataset, now let's do them step-by-step.**

<br>

#### Importing necessary packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    import os
    import seaborn as sns
    sns.set()

#### Reading csv file containing training datadata
    train_df = pd.read_csv("nih/train-small.csv")
#### Printing first 5 rows
    print(f'There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in this data frame')
    train_df.head()
<br>
<p align="center"><img src="Images/1.png"></p>
<br>


## Data types and null values check

#### Look at the data type of each column and whether null values are present
    train_df.info()

    Output:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 16 columns):
    Image                 1000 non-null object
    Atelectasis           1000 non-null int64
    Cardiomegaly          1000 non-null int64
    Consolidation         1000 non-null int64
    Edema                 1000 non-null int64
    Effusion              1000 non-null int64
    Emphysema             1000 non-null int64
    Fibrosis              1000 non-null int64
    Hernia                1000 non-null int64
    Infiltration          1000 non-null int64
    Mass                  1000 non-null int64
    Nodule                1000 non-null int64
    PatientId             1000 non-null int64
    Pleural_Thickening    1000 non-null int64
    Pneumonia             1000 non-null int64
    Pneumothorax          1000 non-null int64
    dtypes: int64(15), object(1)
    memory usage: 125.1+ KB

## Unique IDs check

***PatientId* has an identification number for each patient. One thing we'd like to know about a medical dataset like this is if we're looking at repeated data for certain patients or whether each image represents a different person.**

    print(f"The total patient ids are {train_df['PatientId'].count()}, from those the unique ids are {train_df['PatientId'].value_counts().shape[0]}")

    Ouput :
    The total patient ids are 1000, from those the unique ids are 928 

**As we can see, the number of unique patients in the dataset is less than the total number so there must be some overlap. For patients with multiple records, we'll want to make sure they do not show up in both training and test sets in order to avoid data leakage.**


## Exploring data labels

#### Running the next two code cells to create a list of the names of each patient condition or disease.

    columns = train_df.keys()
    columns = list(columns)
    print(columns)

    Output:
    ['Image', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'PatientId', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


#### Removing unnecesary elements
    columns.remove('Image')
    columns.remove('PatientId')
#### Getting the total classes
    print(f"There are {len(columns)} columns of labels for these conditions: {columns}")

    Output:
    There are 14 columns of labels for these conditions: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

#### Printing out the number of positive labels for each class
    for column in columns:
        print(f"The class {column} has {train_df[column].sum()} samples")

    Output:

    The class Atelectasis has 106 samples
    The class Cardiomegaly has 20 samples
    The class Consolidation has 33 samples
    The class Edema has 16 samples
    The class Effusion has 128 samples
    The class Emphysema has 13 samples
    The class Fibrosis has 14 samples
    The class Hernia has 2 samples
    The class Infiltration has 175 samples
    The class Mass has 45 samples
    The class Nodule has 54 samples
    The class Pleural_Thickening has 21 samples
    The class Pneumonia has 10 samples
    The class Pneumothorax has 38 samples
 
**Have a look at the counts for the labels in each class above. Does this look like a balanced dataset?**
 
**No because the samples are not uniform in the dataset.**

## Data Visualization

#### Extracting numpy values from Image column in data frame
    images = train_df['Image'].values

#### Extracting 9 random images from it
    random_images = [np.random.choice(images) for i in range(9)]

#### Location of the image dir
    img_dir = 'nih/images-small/'
 
    print('Display Random Images')

#### Adjusting the size of our images
    plt.figure(figsize=(20,10))

#### Iterating and plotting random images
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(os.path.join(img_dir, random_images[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
#### Adjustting subplot parameters to give specified padding
    plt.tight_layout()   
<br>
<p align="center"><img src="Images/2.png"></p>
<br>


## Investigating a single image

#### Getting the first image that was listed in the train_df dataframe
    sample_img = train_df.Image[0]
    raw_image = plt.imread(os.path.join(img_dir, sample_img))
    plt.imshow(raw_image, cmap='gray')
    plt.colorbar()
    plt.title('Raw Chest X Ray Image')
    print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
    print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
    print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")


<br>
<p align="center"><img src="Images/3.png"></p>
<br>


## Investigating pixel value distribution

#### Plotting a histogram of the distribution of the pixels
    sns.distplot(raw_image.ravel(), 
                label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}', kde=False)
    plt.legend(loc='upper center')
    plt.title('Distribution of Pixel Intensities in the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# Pixels in Image')
<br>
<p align="center"><img src="Images/4.png"></p>

## Image Preprocessing in Keras

**Before training, we'll first modify our images to be better suited for training a convolutional neural network. For this task we'll use the Keras ImageDataGenerator function to perform data preprocessing and data augmentation.**
<br>

#### Importing data generator from keras
    from keras.preprocessing.image import ImageDataGenerator
    
    Output:
    Using TensorFlow backend.

#### Normalizing images
    image_generator = ImageDataGenerator(
        samplewise_center=True, #Set each sample mean to 0.
        samplewise_std_normalization= True # Divide each input by its standard deviation
    )


## Standardizationing pixel values
**The image_generator we created above will act to adjust our image data such that the new mean of the data will be zero, and the standard deviation of the data will be 1.**
 
**In other words, the generator will replace each pixel value in the image with a new value calculated by subtracting the mean and dividing by the standard deviation.**

<br>
<p align="center"><img src="Images/1.gif"></p>

## Pre-processing our data using the image_generator. 

**In this step we will also be reducing the image size down to 320x320 pixels.**
<br>

#### Flow from directory with specified batch size and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=train_df,
            directory="nih/images-small/",
            x_col="Image", # features
            y_col= ['Mass'], # labels
            class_mode="raw", # 'Mass' column should be in train_df
            batch_size= 1, # images per batch
            shuffle=False, # shuffle the rows or not
            target_size=(320,320) # width and height of output image
    )

    Output:
    Found 1000 validated image filenames.


#### Plot a processed image

    sns.set_style("white")
    generated_image, label = generator.__getitem__(0)
    plt.imshow(generated_image[0], cmap='gray')
    plt.colorbar()
    plt.title('Raw Chest X Ray Image')
    print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
    print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
    print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
<br>
<p align="center"><img src="Images/5.png"></p>

## Seeing a comparison of the distribution of pixel values in the new pre-processed image versus the raw imagez

#### Including a histogram of the distribution of the pixels
    sns.set()
    plt.figure(figsize=(10, 7))

#### Plotting histogram for original iamge
    sns.distplot(raw_image.ravel(), 
                label=f'Original Image: mean {np.mean(raw_image):.4f} - Standard Deviation {np.std(raw_image):.4f} \n '
                f'Min pixel value {np.min(raw_image):.4} - Max pixel value {np.max(raw_image):.4}',
                color='blue', 
                kde=False)

#### Plotting histogram for generated image
    sns.distplot(generated_image[0].ravel(), 
                label=f'Generated Image: mean {np.mean(generated_image[0]):.4f} - Standard Deviation {np.std(generated_image[0]):.4f} \n'
                f'Min pixel value {np.min(generated_image[0]):.4} - Max pixel value {np.max(generated_image[0]):.4}', 
                color='red', 
                kde=False)

#### Placing legends
    plt.legend()
    plt.title('Distribution of Pixel Intensities in the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# Pixel')
<br>
<p align="center"><img src="Images/6.png"></p>
<br>
