# Truck Backer Upper 

## Simulation Before Training 

<img src="videos/simulation-before-training.gif" width="75%" alt="Simulation After Training">

## Simulation After Training

<img src="videos/simulation-after-training.gif" width="75%" alt="Simulation After Training">

## Simulation Beyond Training Boundaries

<img src="videos/simulation-beyond-training.gif" width="75%" alt="Simulation After Training">

## Trajectories

![Trajectory 1](trajectories/lesson-11/trajectory-1.png)

![Trajectory 4](trajectories/lesson-11/trajectory-4.png)

![Trajectory 5](trajectories/lesson-11/trajectory-5.png)

## Training Process 

![Training Process](figures/training-process.png)

## Run the Simulation

To train or test the model, run:

```bash
python truck-backer-upper.py \
    --train_eval eval \
    --final_cab_angle_range -120 120 \
    --final_cab_trailer_angle_diff_range -45 45 \
    --final_x_cab_range 10 35 \
    --final_y_cab_range -7 7 \
    --env_x_range 0 40 \
    --env_y_range -10 10 \
    --draw_trajectory True \
    --num_lessons 10 \
    --truck_speed -0.1 \
    --wandb_log False \
    --save_computational_graph False 
```

## Notes
- The version of the Python that is used in this notebook is 3.10.15. 