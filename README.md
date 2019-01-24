
# Simple Navigagtion with DRL
## How to Solve Simple Navigation Problem by DRL Value-based Method(DQN)
---
We will introduce to solve simple navigation problem by Deep Reinforcement Learning Value-based Method like DQN with Unity ML-Agents Banana environment.

### Prerequisites

#### Theoretical things
* Read about Traditional Reinforcement Learning :  [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto - Second Edition](http://incompleteideas.net/book/the-book.html)
* Read this [scientific article](https://www.cs.swarthmore.edu/~meeden/cs63/s15/nature15a.pdf) that describes Deep Q-Networks.
* Read the [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) that first introduced the Deep Q-Learning algorithm.
* Learn more about Deep Q-Learning and Google DeepMind by watching this [video](https://www.youtube.com/watch?v=xN1d3qHMIEQ).

#### Technical things
* Read about Deep Learnming with PyTorch : https://github.com/udacity/DL_PyTorch

### Project Details
For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.

![Banana](assets/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.



#### Note
The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

We are required to work with the environment that provided as part of the project.

---

### Getting Started
Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

#### Step 1: Clone the DRLND Repository
If you haven't already, please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

#### Step 3: Explore the Environment
After you have followed the instructions above, open Navigation.ipynb (located in the p1_navigation/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.

Watch [this video](https://youtu.be/ltz2GhFv04A) to see what kind of output to expect from the notebook, if everything is working properly!

In the last code cell of the notebook, you'll learn how to design and observe an agent that always selects random actions at each timestep. Your goal in this project is to create an agent that performs much better!

#### (Optional) Build your Own Environment
For this project, we have built the Unity environment for you, and you must use the environment files that we have provided.

If you are interested in learning to build your own Unity environments after completing the project, you are encouraged to follow the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md), which walk you through all of the details of building an environment from a Unity scene.

---

### Instructions
To setup our project environment to run the code in this repository, follow the instructions below.

1. Clone this repository
```
git clone https://github.com/parksurk/drl_value-based_methods.git
```
2. Create (and activate) a new environment with Python 3.6.
    * Linux or Mac:
```
conda create --name drlnd python=3.6
```
    * Windows:
```
conda create --name drlnd python=3.6
activate drlnd
```
3. Follow the instructions in this repository to perform a minimal install of OpenAI gym. (Skip if you done already)
    * Next, install the classic control environment group by following the instructions here.
    * Then, install the box2d environment group by following the instructions here.
4. Clone the repository (if you haven't already!), and navigate to the python/ folder. Then, install several dependencies. (Skip if you done already)
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
5. Create an IPython kernel for the drlnd environment. (Skip if you done already)
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
6. Run Jupyter Notebook
```
jupyter notebook
```
7. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.
![Jupyter Notebbok Kernel Setting](assets/jupyter_notebook_kernel_menu.png)
8. Click **Report.ipynb** on root directory
