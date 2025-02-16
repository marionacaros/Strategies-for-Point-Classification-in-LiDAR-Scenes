## Effective Strategies for Point Classification in LiDAR Scenes

Created by [Mariona Carós](https://www.linkedin.com/in/marionacaros/), from University of Barcelona (UB) and [Cartographic Institute of Catalonia (ICGC)](https://www.icgc.cat/es), [Santi Seguí](https://ssegui.github.io/) and [Jordi Vitrià](https://algorismes.github.io/) from UB and [Ariadna Just](https://www.linkedin.com/in/ariadna-just-0a667559/?originalSubdomain=es) from ICGC.

This work is based on our [paper](https://www.mdpi.com/2072-4292/16/12/2153).

### Abstract

Light Detection and Ranging systems serve as robust tools for creating three-dimensional maps of the Earth's surface, represented as point clouds. Point cloud scene segmentation is essential in a range of applications aimed at understanding the environment, such as infrastructure planning and monitoring. However, automating this process encounters notable challenges due to variable point density across scenes, ambiguous object shapes, and substantial class imbalance. Consequently, manual intervention remains prevalent in point classification to address these complexities. In this work, we study the elements contributing to the automatic semantic segmentation process with deep learning, conducting empirical evaluations on a self-captured dataset by a hybrid airborne laser scanning sensor combined with two nadir cameras in RGB and near-infrared over a terrain characterized by hilly topography and dense forest cover. Our findings emphasize the importance of employing appropriate training and inference strategies to achieve accurate classification of data points across all categories. The proposed methodology not only facilitates the segmentation of varying size point clouds but also yields a significant increase in Intersection over Union score when compared to preceding methodologies.

### Predictions

![plot](figs/preds_terra_alta.png)
![plot](figs/preds_OOD.png)


