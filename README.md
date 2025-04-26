# Truck Backer Upper 

## Simulation Before Training 

<img src="videos/simulation-before-training.gif" width="75%" alt="Simulation After Training">

## Simulation After Training

<img src="videos/simulation-after-training.gif" width="75%" alt="Simulation After Training">

## Simulation Beyond Training Boundaries

<img src="videos/simulation-beyond-training.gif" width="75%" alt="Simulation After Training">

## First 5 Trajectories

![Trajectory 1](trajectories/lesson-11/trajectory-1.png)

![Trajectory 2](trajectories/lesson-11/trajectory-2.png)

![Trajectory 2](trajectories/lesson-11/trajectory-3.png)

![Trajectory 4](trajectories/lesson-11/trajectory-4.png)

![Trajectory 5](trajectories/lesson-11/trajectory-5.png)

![Trajectory 6](trajectories/lesson-11/trajectory-6.png)

![Trajectory 7](trajectories/lesson-11/trajectory-7.png)

![Trajectory 8](trajectories/lesson-11/trajectory-8.png)

![Trajectory 9](trajectories/lesson-11/trajectory-9.png)

![Trajectory 10](trajectories/lesson-11/trajectory-10.png)

## Run the Simulation

Create and activate the conda environment with:

```bash
conda env create -f conda_env.yaml
``` 

```bash
conda activate truck_backer_upper
```

To train models, run:

```bash
python truck-backer-upper.py \
    --train_test train \
    --final_cab_angle_range -120 120 \
    --final_cab_trailer_angle_diff_range -45 45 \
    --final_x_cab_range 10 35 \
    --final_y_cab_range -7 7 \
    --env_x_range 0 40 \
    --env_y_range -10 10 \
    --num_lessons 10 \
    --truck_speed -0.1 \
    --wandb_log False \
    --save_computational_graph False 
```

To test the models, run:

```bash
python truck-backer-upper.py \
    --train_test test \
    --final_cab_angle_range -120 120 \
    --final_cab_trailer_angle_diff_range -45 45 \
    --final_x_cab_range 10 35 \
    --final_y_cab_range -7 7 \
    --env_x_range 0 40 \
    --env_y_range -10 10 \
    --draw_trajectory True \
    --truck_speed -0.1 \
```