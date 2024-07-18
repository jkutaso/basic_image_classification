# basic_image_classification
I want to practice making a neural net so let's try to do image classification

# First working model

I got a model working

def initialize_cnn():
    input_shape = (32, 32, 3)

    # Create a Sequential model
    model = models.Sequential()
    
    # Add the input layer with the defined input shape
    model.add(layers.Input(shape=input_shape))
    
    # Add convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model

I trained in 10 epochs with batch size of 32. Test accuracy was about 72%.

# Testing different parameters

Iterating over different pooling type, different widths of the convolution window and training for 30 epochs with a snapshot after each 10:
![image](https://github.com/user-attachments/assets/4b7f4de8-363c-4f3e-b43b-557ade7e2268)

Grouping by epochs:

![image](https://github.com/user-attachments/assets/9458f959-bf4c-4d7c-a0d9-fdf53d1259aa)

We see that train accuracy keeps getting better but in fact test loss increases while test accuracy stays the same. Looks like we are overfitting, although perhaps if we run it for longer it will grok eventually. Perhaps it is losing confidence on the ones it is getting right, although it makes more sense that it is gaining confidence from training longer, so probably it is more confident on the ones it gets wrong.

Grouping by pooling type:

![image](https://github.com/user-attachments/assets/b92626c5-f672-41a5-a5e3-f5948ec4cb68)

It looks like the results are more or less the same.

Grouping by conv_width:


![image](https://github.com/user-attachments/assets/b0bad5d3-53c1-4dab-b2a7-96cc6101a95b)

It looks like smaller windows do a better job, so let's stick with that.

# Adding batch normalization and dropout

After adding batch normalization and dropout, the training was much slower for a small improvement. We were able to get as high as 80% test accuracy
