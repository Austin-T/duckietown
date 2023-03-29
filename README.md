# Duckietown
Welcome to the home of my CMPUT 412 (Experimental Mobile Robotics) projects. This repository contains code that will be deployed on autonomous vehicles known as "duckiebots". This course builds on top of the [Duckietown](https://www.duckietown.org/) learning platfrom, serving as an introduction to the fundamentals of autonomous robotics.

# Repository structure
At the root-level of this repository, you will notice that there are a set of folders coresponding to each "exercise" that I completed as part of the CMPUT 412 curriculum. Many exercises will be further subdivided into "units." Many folders will contain README files to better explain their contents.

Any code that is intended to be run by others will have instructions for doing so.

# Contributors
Code within the [exercise_2](exercise_2/) and [exercise_3](exercise_3/) subfolder was developed jointly with [MoyinF](https://github.com/MoyinF).

# Exercises Overview
## Exercise 1: Duckiebot Assembly and Basic Development
In this exercise I learned the basic knowledge and skills needed to operate a
Duckiebot, including robot assembly, networking, scripting, containerization, version control, and more.

The [exercise_1](exercise_1/) directory contains code that spans a set of projects (B-2, B-5, and B-6) from the book *Hands On Robotics Development using Duckietown.*

## Exercise 2: Ros Development and Kinematics
In this exercise, I learned about different components of ROS (Nodes, Topics, Services, Bags, etc) and the basic applications of Robotic Kinematics and Odometry.

The [exercise_2](exercise_2/) directory contains a ROS docker image.

# Exercise 3: Computer Vision for Robotics
This exercise provided an introduction to computer vision and localization in robotics. It builds on top of the deadreckoning techniques developed in exercise 2, using fiducial markers to enable better pose estimation in the environment.

The [exercise_3](exercise_3/) directory contains a ROS docker image.

# Exercise 4: Don't Crash! Tailing Behaviour
This mini exercise involved the use of PID controllers for lane following, vehicle tailing, and collision avoidance.

The [exercise_4](exercise_4/) directory contains a ROS docker image.

# Exercise 5: ML for Robotics
This lab provides an introduction to machine learning in robotics. The object of the lab is to detect hand-written digits posted above fiducial markers.

The [exercise_5](exercise_5/) directory contains two ROS docker image; one is meant to be run on a duckiebot while the other should be run on a remote computer.
