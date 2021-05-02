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

![GridBasedHeatmap](https://user-images.githubusercontent.com/26316298/116830954-55db7800-ab72-11eb-9079-b235be55cacd.jpg)

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



### Compressed Videos

### Metrics

## Future Directions

### Machine Learning
