## Truck Kinematics 

![Truck](figures/truck-kinematics.png)

## Training Emulator 

![Truck](figures/emulator-training.png)

## Training Controller

![Truck](figures/controller-training.png)

## Simulation Before Training 

<img src="gifs/lesson-0-2025-06-01_01-49AM.gif" alt="Simulation Before Training" width="600">

## Simulation After Training

<img src="gifs/lesson-10-2025-06-01_01-51AM.gif" alt="Simulation After Training">


## Trajectories

![Trajectory 1](trajectories/lesson-10/trajectory-1.png)

![Trajectory 2](trajectories/lesson-10/trajectory-2.png)

![Trajectory 2](trajectories/lesson-10/trajectory-3.png)

![Trajectory 4](trajectories/lesson-10/trajectory-4.png)

![Trajectory 5](trajectories/lesson-10/trajectory-5.png)

![Trajectory 6](trajectories/lesson-10/trajectory-6.png)

![Trajectory 7](trajectories/lesson-10/trajectory-7.png)

![Trajectory 8](trajectories/lesson-10/trajectory-8.png)

![Trajectory 9](trajectories/lesson-10/trajectory-9.png)

![Trajectory 10](trajectories/lesson-10/trajectory-10.png)

## Run the Simulation


Create and activate the conda environment with:

```bash
conda env create -f conda_env.yaml
``` 

```bash
conda activate truck_backer_upper
```

To test the models inside the training region, run:

```bash
python truck-backer-upper.py
```
To test the models outside the training region, run:

```bash
python truck-backer-upper.py\
    --env_x_range 0 100\
    --env_y_range -30 30\
    --test_x_cab_range 40 90\
    --test_y_cab_range -20 20
```

To train both emulator and controller models, run:

```bash
python truck-backer-upper.py --train_emulator True 
```

To train only controller models, run:

```bash
python truck-backer-upper.py --train_controller True 
```