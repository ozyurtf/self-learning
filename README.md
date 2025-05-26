# Truck Backer Upper 

## Truck Kinematics 

![Truck](figures/truck.png)

![Truck](figures/kinematics.png)

## Training Emulator 

![Truck](figures/emulator-training.png)

## Training Controller

![Truck](figures/controller-training.png)


## Simulation Before Training 

<img src="videos/simulation-before-training.gif" width="75%" alt="Simulation After Training">

## Simulation After Training

<img src="videos/simulation-after-training.gif" width="75%" alt="Simulation After Training">

## Simulation Beyond Training Boundaries

<img src="videos/simulation-beyond-training.gif" width="75%" alt="Simulation After Training">

## First 10 Trajectories

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
    --test_x_cab_range 70 90\
    --test_y_cab_range -15 15\
    --env_x_range 0 100
```



To train both emulator and controller models, run:

```bash
python truck-backer-upper.py\
    --train_emulator True 
```

To train only controller models, run:

```bash
python truck-backer-upper.py\
    --train_controller True 
```