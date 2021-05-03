# vr-user-behavior-analysis


Research Poster: [Link](https://raw.githubusercontent.com/ACM-Research/vr-user-behavior-analysis/main/ACM%20Research%20Poster%20-%20VR.pdf)

![Poster picture](https://github.com/ACM-Research/vr-user-behavior-analysis/blob/5c57497fe9c28c91c7e6b9b0f5d829128d4305f8/ACM%20Research%20Poster%20-%20VR.png)
## Repository Structure

## Introduction
Streaming a 4k virtual reality (VR) video requires about 600 Mbit/s internet speed connection. The average user does not have access to these speeds. Our proposed solution was to adaptively render each frame of a VR video based on where users are looking. This would ideally reduce file size while having no impact on a user's experience.

### Hypothesis
If we discard areas from the frame that are not seen by the user and compress remaining areas based on relative values of importance, viewers will notice a negligible loss of quality of experience while significantly reducing file size.

## Timeline

![Timeline](https://user-images.githubusercontent.com/26316298/116831271-d8186c00-ab73-11eb-95a5-8569c14f1d25.PNG)

### Heatmaps

We began our research by developing heatmaps. At first, we did this by splitting frames into square regions and assigning heat values based on the centers of viewports. The heatmaps created this way were simple. However, they did not accurately define how attention is distributed throughout the frame in a continuous sense.

![GridBasedHeatmap](https://user-images.githubusercontent.com/26316298/116834158-672c8080-ab82-11eb-9677-c6fa6de3558a.jpg)

This issue was fixed with gradient heatmaps. These heatmaps were developed by stacking viewport ellipses of varying heat for each user's center of vision. Each user's focus decays outward from the center, as it is not possible to focus on multiple areas at the same instant.

![GradientHeatmap](https://user-images.githubusercontent.com/26316298/116830958-5a079580-ab72-11eb-8b94-ae56af6392d0.png)

### Voting Functions

To model how focus decays from the center of viewports, we created 4 voting functions. The square root and parabolic voting functions were eventually dropped, as they were too extreme to find a balance between user experience and storage saved.

#### Linear
![linear](https://user-images.githubusercontent.com/26316298/116831167-51fc2580-ab73-11eb-962a-c58e2e87872f.png)

#### Semicircle

![semicircle](https://user-images.githubusercontent.com/26316298/116831224-9e476580-ab73-11eb-90c0-28fcd85a1a24.png)

#### Square Root

![squareroot](https://user-images.githubusercontent.com/26316298/116831238-ae5f4500-ab73-11eb-9cf1-3cb5444dffee.png)

#### Parabolic

![parabolic](https://user-images.githubusercontent.com/26316298/116831243-b6b78000-ab73-11eb-8f04-ca133e0ea386.png)

### Resolution Maps

With our improved heatmaps, we could develop resolution maps. Resolution maps are "rounded" heatmaps in which we make the following assumptions:
* A user can only focus on the central 80% of their full field of view
* If there is no heat, we can guarantee nobody is looking in that area. We can color this area black. This is resolution level 0.
* If the heat falls below 20% of the maximum in a given frame, we can treat it as an area of low focus and render it in a lower resolution. This is resolution level 1.
* If the heat is above this threshold, we should render it in high resolution. This is resolution level 2.

For example, the heatmap 

![GradientHeatmap](https://user-images.githubusercontent.com/26316298/116831497-650ff500-ab75-11eb-8faf-dca67e0e6a09.jpg)

is translated to the resolution map

![ResolutionMap](https://user-images.githubusercontent.com/26316298/116834163-70b5e880-ab82-11eb-9bcd-8f779c413a45.png)

### Compressed Videos

Using resolution maps, we can compress video frames. Frames are split into several smaller images, which each represent one distinct resolution level. From there, each partial image is compressed to its resolution level (see above). These compressed partial images are then recombined into a full, smartly compressed frame. If desired, we can combine these compressed frames into a compressed video.

### Metrics

We defined the storage statistic as the ratio of the compressed image’s size as compared to the original. Taking the central 80% of a user’s viewport as “important”, we similarly defined the “user experience rating” as the percentage that “important” area that is rendered in full resolution. 

We found that, in the case of video 23 with the linear voting function, we could achieve a storage statistic of 32.5% while keeping the user experience rating at 78.6%. The semicircle voting function achieved an even lower storage statistic of 31.0% but sacrificed a significant amount of user experience (new user experience of 60.3%) to do so. 

#### Storage vs User Experience per Frame (Linear Voting Function, Video 23)
![compresslinear23PerFrameStats](https://raw.githubusercontent.com/ACM-Research/vr-user-behavior-analysis/main/compresslinear23PerFrameStats.gif)

#### Storage vs User Experience per Frame (Semicircle Voting Function, Video 23)
![image](https://raw.githubusercontent.com/ACM-Research/vr-user-behavior-analysis/main/compresssemiCrcl23PerFrameStats.gif)

### Machine Learning

The methods used in this project rely on having a large amount of viewport data. Because this may not be the case in many applications, we tested the feasibility of using machine learning models to predict where users are looking based on a smaller amount of data.

Our current [Kaggle](https://www.kaggle.com/rishivilla/vr-user-analysis-model) model predicted the simplified heatmap

![Predicted (3)](https://user-images.githubusercontent.com/26316298/116834174-7e6b6e00-ab82-11eb-880c-9bec0cc45bbd.png)

when the actual simplified heatmap for the frame in question was

![Actual (2)](https://user-images.githubusercontent.com/26316298/116834176-83c8b880-ab82-11eb-8510-b2e82ddece51.png)

Although the model is not perfect, it serves as a strong proof-of-concept for the use of machine learning to compensate for a lack of large user datasets.

## Future Directions

There are a number of ways to further this research.
* Machine Learning
  * Further optimize model
  * Include facial recognition and computer vision algorithms in model
* Divide users into clusters based on common viewing behavior
* Account for auditory influences in analyzing user viewing behavior
* Collaborate with optometrists to create voting functions with a greater biological basis

## Contributors

* Varin Sikand
* Ryan Aspenleiter
* Shreyon Roy
* Rishi Villa 
* Sunny Guan - Team Supervisor
* Dr. Ravi Prakash - Research Advisor



