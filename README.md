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

