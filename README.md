# Machine_Learning_Assign_4
The forth assignment.

The python edition is 3.6.6

numpy - 1.15.4

keras -	2.2.4

Tensorflow - 1.12.0 

PIL - 5.3.0

matplotlib - 3.0.0

scipy - 1.1.0

The normal model is in "VGG11.py" file, with most functions used throughout the exercise.

The main function contains the hyperparameters including image size, batch size, number of classes, epochs.
It also contains the process of training the model with SGD and MNIST dataset, and store the log into "hist" variable.

The last 3 lines are the codes that will generate output for the questions.
If you execute "plot_result()" function, it will generate 4 figures of accuracy/loss on test/training set within 5 epochs.
If you execute "plot_rotate()" function, it will generate a figure of accuracy with different degree of rotation on the test set.
If you execute "plot_blur()" function, it will generate a figure of accuracy with different radius of blurring on the test set.

The model with regularization is in "VGG11_Reg.py" file, and most of the functions used here are imported from "VGG11.py", and the "plot" functions will generate same figures with a suffix of "_Reg".


The model with data augmentation is in "VGG11_Aug.py" file, also, most of the functions used here are imported from "VGG11.py", and the "plot" functions will generate same figures with a suffix of "_Aug".
