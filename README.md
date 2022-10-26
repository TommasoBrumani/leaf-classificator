# Leaf Classificator Artificial Neural Network
## Overview
The program is a `Python` implementation using `Jupyter Notebook` of an artificial neural network model for the classification of leaf images into different categories, employing a variety of data-augmentation and deep learning techniques. 

### Authors
<b>DEEPressi</b> Team
- <b>Tommaso Brumani</b> (tommaso.brumani@mail.polimi.it)
- <b>Riccardo Pazzi</b> (riccardo.pazzi@mail.polimi.it)
- <b>Gianluca Ruberto</b> (gianluca.ruberto@mail.polimi.it)

### License
The project was carried out as part of the 2021/2022 '<b>Artificial Neural Networks and Deep Learning</b>' course at <b>Politecnico of Milano</b>, where it was evaluated based on its accuracy over a test set and awarded a score of 5/5.

## Project Specifications
The project consisted of producing a model capable of classifying pictures of specific plant leaves with sufficient accuracy, with the provided training dataset containing unbalanced representations of each class.

The team was granted great freedom in choosing its approach, but encouraged to use the techniques covered in the course.

The iterative process of improving the model (documented in the project report) entailed both dataset manipulation and model experimentation.

In regards to the dataset provided, the team employed data augmentation, regularization, and oversampling to improve the available material.

In regards to the neural network structure the team begun by testing several network shapes and sizes, including both the use of convolutional and dense layers, but eventually decided to make use of transfer learning with fine tuning as it provided the best results (the final network being transferred from the EfficientNetB7 model). 

## Folder Structure
* `report`: the report detailing the various approaches that were tried as part of the development of the model
* `src`: the `jupyter notebook` code for the final model, as well as some previous attempts