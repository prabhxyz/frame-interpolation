
# Creating a Frame Interpolation Model from Scratch — using Convolutional Fusion Upsampling

[![YouTube Video](https://img.youtube.com/vi/6NDgmQptTZY/maxresdefault.jpg)](https://www.youtube.com/watch?v=6NDgmQptTZY)

<p align="center">Click the YouTube Video Above for More Information</p>

## Introduction 

Frame interpolation, also known as ‘inbetweening’ can allow you to generate frames between pairs of 2 intermediate frames.

Although seemingly simple at first — the applications are promising.

**Motion Analysis:**

* **Sports science:** Frame interpolation can be used to analyze the movements of athletes in order to identify areas for improvement. For example, it can be used to track the motion of a runner’s feet to see if they are landing correctly, or to track the rotation of a golfer’s swing to see if they are using the correct technique.

* **Security and surveillance:** Frame interpolation can be used to analyze video footage of security cameras to identify potential threats or suspicious activities. For example, it can be used to track the movement of a person in a crowd to see if they are acting suspiciously, or to track the movement of a vehicle to see if it is following someone.

**Fluid Dynamics:**

* **Aerodynamics:** Frame interpolation can be used to study the flow of air around aircraft and other objects. This can help engineers to design more efficient and aerodynamic vehicles.

* **Hydrodynamics:** Frame interpolation can be used to study the flow of water and other fluids. This can help engineers to design more efficient and effective pumps, turbines, and other fluid machinery.

**Chemical Reactions:**

* **Combustion:** Frame interpolation can be used to study the combustion of fuels. This can help scientists to develop more efficient and environmentally friendly engines.

* **Polymerization:** Frame interpolation can be used to study the polymerization of plastics and other materials. This can help scientists to develop new materials with improved properties.

**Material Science:**

* **Fracture mechanics:** Frame interpolation can be used to study the fracture of materials. This can help engineers to design materials that are more resistant to cracking and failure.

* **Fatigue:** Frame interpolation can be used to study the fatigue of materials. This can help engineers to design materials that are more resistant to wear and tear.

**Biological Studies:**

* **Cell biology:** Frame interpolation can be used to study the movement of cells and other biological structures. This can help scientists to understand the processes of cell division, migration, and differentiation.

* **Neuroscience:** Frame interpolation can be used to study the activity of neurons in the brain. This can help scientists to understand the processes of learning, memory, and perception.

and there are also various other applications in entertainment and production. This repo consists of code used to create a entire frame interpolation model, all from scratch — and how I managed to do it.

![](https://cdn-images-1.medium.com/max/3840/1*zwyTPbu93XlXxBwdbWTdyw.png)

## Gathering the Dataset

Training the frame interpolation model requires a diverse dataset of frames from various videos. When learning from a dataset like such, the model attempts to learn the complex relationships between consecutive frames to accurately synthesize the missing intermediate frames.

Specifically the model learns…

* the motion patterns,

* the appearance changes,

* different lighting conditions,

* and various other features.

There are various popular datasets that the frame interpolation model can learn from.

* **Vimeo-90K:** This is a large dataset of high-quality videos with varying frame rates and motion patterns. It is commonly used for evaluating frame interpolation models.

* **Middlebury:** This dataset contains a collection of video sequences with ground truth optical flow between consecutive frames. It is useful for training and evaluating optical flow-based frame interpolation methods.

* **Adobe 2K Training Set:** This dataset consists of 2K resolution videos with GoPro-style camera motion. It is suitable for training frame interpolation models that handle dynamic scenes.

* **DAVIS:** This dataset includes high-quality videos with complex motion and occlusion. It is a challenging benchmark for evaluating frame interpolation methods.

* **EPFL:** This dataset provides a collection of videos with extreme motion, such as sports and action scenes. It is useful for training models that can handle large displacements between frames.

![Vimeo-90k Dataset](https://cdn-images-1.medium.com/max/2000/1*wes4V_H6LFW_ZNvN9kQjJA.gif)

To create a dataset from scratch you can download stock videos from [pexels.com](https://www.pexels.com/), and apply pre-processing to create a dataset. In the following code, all the videos are saved into ‘dataset/videos’. The videos have to be specifically 720p to have proper ratios, although that can be easily changed.

## Training the Model

This approach to a frame interpolation model utilizes a convolutional neural network (CNN) architecture to learn the complex relationships between input frames and the corresponding missing intermediate frame. The CNN extracts features from the input frames and combines them in a hierarchical manner to effectively capture the temporal dynamics of video sequences.

**Model Architecture:**
The FrameInterpolationModel class represents the core neural network architecture. It comprises three main components:

 1. **Feature Extractors:** A series of three feature extractor blocks, each consisting of two convolutional layers followed by a ReLU activation function, responsible for extracting high-level features from the input frames.

 2. **Upsampling:** A bilinear upsampling layer that scales the extracted features to match the resolution of the target intermediate frame.

 3. **Fusion and Upsampling Convolutional Layers:** A fusion convolutional layer that combines the upsampled feature maps and a final upsampling convolutional layer that generates the predicted intermediate frame.

**Data Preparation:**
The FrameInterpolationDataset class handles the preparation of training data from a collection of video sequences.

 1. The constructor initializes the dataset with the root directory containing the video frames and defines a transformation function to apply pre-processing steps if necessary.

 2. The __len__ method returns the total number of data samples, while the __getitem__ method retrieves a specific data sample consisting of two input frames (frame1 and frame2) and the corresponding target intermediate frame.

**Model Training:**
The main training loop involves several steps:

1. **Model and Loss Definition:** The model instance, loss function (mean squared error), and optimizer (Adam) are initialized.

2. **Dataset Loading:** The training dataset is loaded using a data loader that batches the data efficiently.

3. **Training Epochs:** A loop iterates over a specified number of epochs (training cycles).

4. **Mini-batch Processing:** Within each epoch, the loop iterates over mini-batches of data.

5. **Input Preparation:** Input frames are concatenated and sent to the model.

6. **Output Generation:** The model generates the predicted intermediate frame.

7. **Target Upsampling:** The target intermediate frame is upsampled to match the predicted frame’s resolution.

8. **Loss Calculation:** Mean squared error is computed between the predicted and target frames.

9. **Backpropagation and Optimization:** Gradients are calculated, and the optimizer updates the model’s parameters to minimize the loss.

10. **Loss Reporting:** The loss value is reported periodically to track the training progress.

## The Model Performance

![](https://cdn-images-1.medium.com/max/2560/1*Am2Ff-4J_oVoiX-REyID1A.png)

After various efforts to train the model on more epoch, the maximum my computer was able to manage in a reasonable amount of time (~5 hours) was 10 epochs.

Although the model and the results are premature- if trained on a dedicated GPU with at least 20 epochs, the model will be able to create seamless intermediate frames.

## Interesting Resources

*Research Papers:*

* “Deep Video Interpolation” by Shi et al. (2016): This paper proposes a deep learning-based approach to frame interpolation using a convolutional encoder-decoder architecture. The network takes two input frames and generates an intermediate frame by learning to extract and combine features from the input frames. [https://arxiv.org/abs/2202.04901](https://arxiv.org/abs/2202.04901)

* “EDSR: Enhanced Deep SR for Real-Image Super-Resolution” by Lim and Kim (2017): This paper introduces an enhanced super-resolution network (EDSR) that utilizes residual connections and attention mechanisms to improve the quality of super-resolved images. The EDSR architecture can also be adapted for frame interpolation tasks. [https://arxiv.org/pdf/2112.12089](https://arxiv.org/pdf/2112.12089)

* “Real-Time Single-Image Video Super-Resolution with an Efficient Attention Mechanism” by Wang et al. (2018): This paper proposes a real-time single-image video super-resolution (SR) method using an efficient attention mechanism. The network employs a temporal attention mechanism to capture long-range dependencies between frames, enhancing the quality of interpolated frames. [https://arxiv.org/abs/2210.05960](https://arxiv.org/abs/2210.05960)

*Resources:*

* Frame Interpolation” by Wikipedia: This comprehensive Wikipedia article provides an overview of frame interpolation, including its history, various techniques, and applications. [https://en.wikipedia.org/wiki/Motion_interpolation](https://en.wikipedia.org/wiki/Motion_interpolation)

* “Frame Interpolation: A Survey of Deep Learning Approaches” by Jiang et al. (2020): This survey paper provides an in-depth overview of deep learning-based frame interpolation methods, covering various network architectures, loss functions, and evaluation metrics. [https://arxiv.org/pdf/2302.08455](https://arxiv.org/pdf/2302.08455)

* “Frame Interpolation with CNNs: A Review” by Niklaus et al. (2020): This review paper focuses on convolutional neural network (CNN)-based frame interpolation techniques, discussing the advantages and limitations of different approaches. [https://arxiv.org/abs/1511.08458](https://arxiv.org/abs/1511.08458)
