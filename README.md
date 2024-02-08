# Welcome to your ready-to-go plug and play Machine Teaching framework code-base!

This repository provides a comprehensive foundation for building a Machine Teaching (MT) framework from scratch. Here's what you'll find:
  1) **Abstract Classes:** Abstract classes are provided to define the functionality expected from each component of the MT framework. These abstract classes serve as guides for implementing specific components.
  2) **Component Implementations:** Handy implementations of each component are included to demonstrate how they interact with each other. These implementations offer practical examples of the abstract concepts outlined in the abstract classes.
  3) **Demo Script:** A demo script showcases an end-to-end use case of the MT framework. This script runs seamlessly and provides insights into how all components come together to form a functional MT framework.

**To run this code simply clone the repository and run the `PlugAndPlay.py` script.**

This code-base contains 2 main flows:
## Data preparation flow:
![Data preparation flow](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/Data_prep_concise.png)

This flow runs once per dataset and consists of two main components:
   - **Dataset Download:** This component downloads a dataset into a local folder (e.g., Datasets/Binary_mnist) using the `DatasetDownload.py` script. You can specify the number of images to download.
   - **Feature Extraction:** Here, image representations are created for all images and saved locally for future use. The extracted features and labels are stored in a designated folder (e.g., Dataset_features/Binary_mnist).

## Teaching flow:
![MT flow](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/mt_flow.png)
This flow represents the core MT process, where a teacher instructs a human learner. Key components of this flow are defined in the `AbstractComponents.py` file, which holds abstract classes for each MT component. These abstract classes ensure smooth communication between different components and facilitate modularity and flexibility.
The `Teachers`, `Teaching logics`, and `Virtual Learners` folders contain various implementations of specific MT components, sub-divided occaisonally according to popular trends in the research (i.e., Teaching logics are separated into Data-centric and Learner-centric modules).

Finally, `EvaluationComponent.py` brings everything together by orchestrating the teaching process, collecting metrics, generating visualizations, and saving results to a specified directory.

This project structure, along with modular abstract classes and basic implementations, aims to encourage researchers to contribute their own implementations, fostering a united, reproducible, and flexible research frontier.

A visual guide to the repository:

![Visual Guide](https://raw.githubusercontent.com/orlinle/machine_teaching_framework/master/readme_img/visual_guide.png)
