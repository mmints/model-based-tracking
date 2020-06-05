# Batchelor Thesis: Model-Based Tracking with GPU-Accelerated Particle Filter Estimation and Stereo Vision

Tracking is one of the largest application providers of computer vision and computer graphics.
Model-based tracking is a widespread implementation that has received much attention in recent years.
This bachelor thesis presents an implementation of a model-based tracking system with six degrees of freedom.
For data acquisition, photometric and geometric images of the environment are taken by a ZED stereo camera.
The system is based on the particle filter, an already well-proven algorithm for state estimation. 
Due to the mathematical properties of the filter, parallel calculations can be performed on graphic processors specially designed for such operations.
For this purpose OpenGL and NVIDIAs CUDA were used.
The developed tracking system is then evaluated quantitatively for performance and qualitatively for accuracy.

**Supervisor:** Prof. Dr.-Ing. Stefan Müller, M.Sc. N. Höhner

**Maintainer:** Mark O. Mints
