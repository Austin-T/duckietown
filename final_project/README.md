# Final Project: Autonomous Driving with Duckietown

This repository contains implementation solutions for the final project. For information about the project, please read the [report](https://sites.google.com/ualberta.ca/famobiwo-cmput-412/labs/final-project).

## Execution:

To set the stall parameter, as well as other parameters such as wheel velocities, omega values, stopping distances, etc., ssh into the duckiebot and create a file in the `/data/` folder titled `final_config.yaml`. The exact file we used is in this repository. You can change the numbers in `/data/final_config.yaml`, for example with the following steps:

```
ssh duckie@csc229xx.local # where csc229xx is the duckiebot's hostname
vim /data/final_config.yaml # creates or opens the stall file, where you write the number of the stall and save
```

To run the program, ensure that the variable `$BOT` stores your robot's host name (ie. `csc229xx`), and run the following commands:

```
dts devel build -f -H $BOT.local
dts devel run -H $BOT.local
```

The program shuts down automatically after completing stage 3.
To shutdown the program before that, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the Duckiebot detections starter code by Zepeng Xiao [https://github.com/XZPshaw/CMPUT412503_exercise4](https://github.com/XZPshaw/CMPUT412503_exercise4).

Build on top of by Nadeen Mohamed, Moyinoluwa Famobiwo, and Austin Tralnberg.

Autonomous lane following code was also borrowed from Justin Francis.
