# Welcome to your ready-to-go plug and play Machine Teaching (MT) framework code-base!

This repository has everything you need to build a MT framework from scratch, including:
  1) Helpful abstract classes helping you focus on the functionality each of the components should implement.
  2) Handy implementations of each of the components so you get a feel for how they interact with each other
  3) 1.2.3... Demo! This script is a ready to run. Just press play to get a working demo of existing MT frameworks implemented within the code-base.
     Using this script you can switch different implementations within the code-base and easily evaluate the results of different implementations.

**To run this code simply clone the repository and run the PlugAndPlay.py script.**

This code-base contains 2 main flows:
## Data preparation flow:
![Data preparation flow](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/Data_prep_concise.png)

This flow runs once per dataset and its products can be re-used for future MT flows. The data prepration flow includes two main components:
   - **Dataset Download:** Downloading a dataset into a local folder (e.g., Datasets/Binary_mnist - contains images of 2 mnist classes) _DatasetDownload.py_ is a helper script which downloads a given number of images from keras datasets and saves them locally.
   - **Feature Extraction:** Creating image representations of all the images and saving them locally for future use (e.g., Dataset_features/Binary_mnist contains an array of all the images features and a corresponding array of labels)
This flow needs to be executed once per dataset, and then the features can be re-used by multiple implementations of Teacher/Learner components.

## Teaching flow:
![MT flow](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/mt_flow.png)
This is the actual MT flow, where the teacher attempts to teach a human Learner. The building blocks of the MT flow are to be found within the file  _AbstractComponents.py_ where each MT component is represented by an abstract class, encompassing the methods that must be implemented in order to ensure smooth communication between the different components. _AbstractComponents.py_ serves as an excellent starting point for implementing a MT framework which is cohesive, modular and flexible to different implementations of specific components.
In order to make the code-base easily comprehensive and extendable there are a few implementations of each component which may be found in the corresponding folders: 
- Teachers: this folder creates different implementations of MT Teachers. Following the main trends in MT research, the implementations are divided into sub-modules of Batch Teachers and Interactive Teachers. There are multiple implementations of each, for the purpose of this code-base I provided an example of a Batch Teacher and I leave implementation of Interactive Teachers for future work.
- Teaching logics: this folder holds implementations of the teaching logic utilized by the Teacher. Again, following literature trends I divided the implementations to two modules: Learner-centric and Data-centric. The main difference between them being that the Learner-centric teaching logics require a "Virtual Learner" - which is a representation of the human Learner (see next section). The Learner-centric logic relies on the Virtual Learner during sample choice, wheras Data-centric teaching logics will rely on the feature of the teaching images themselves.
- Virtual Learners: this folder contains computational representations of the human Learner which may by utilized by the Learner-centric teaching logic when performing the teaching sample choice. Since the implementations of this component in the literature and diverse and varied, I opted to create different modules of Virtual Learners according to the archetypes of Learners in the literature: Cognitive Learner, Computational Learner (Gradual Learners) and Global Learners.

Finally, the component that brings it all together and performs the actual teaching process is encompassed in the _Evaluation component_. This essential component holds all the other building blocks and runs the teaching process, gathering metrics along the way, and finishes by creating visualizations and saving them to a given directory (Results). 

In the structure of this project, along with the modular abstract classes and basic implementations, we hope to encourage future MT researchers to add their own implementations, creating a cohesive, united, reproducible and flexible research frontier.



A visual guide to the repository:

![Visual Guide](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/visual_guide.png)
