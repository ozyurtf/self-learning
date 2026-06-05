<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap');

body,
.wrapper,
h1, h2, h3, h4, h5, h6,
p, li, a, td, th, blockquote {
  font-family: 'Avenir Next', 'Avenir', 'Nunito Sans', sans-serif !important;
}

/* Hide the heading anchor-link icons (they render as tofu/hex boxes
   because the forced font has no glyph for the octicon character). */
h1 a.anchor, h2 a.anchor, h3 a.anchor,
h4 a.anchor, h5 a.anchor, h6 a.anchor,
h1 a[aria-hidden], h2 a[aria-hidden], h3 a[aria-hidden],
h4 a[aria-hidden], h5 a[aria-hidden], h6 a[aria-hidden],
.anchor, .octicon, .octicon-link,
.octicon-link::before, .anchor::before, .anchor::after {
  display: none !important;
  content: none !important;
}
</style>

**Note**: I developed this project as part of a challenge project suggested by Dr. Alfredo Canziani in my Deep Learning course at New York University. I took the initial code that sets up a simulated environment from [here](https://github.com/Atcold/NYU-DLSP20/blob/master/14-truck_backer_upper.ipynb). Inspiring by the Truck Backer Upper paper referenced at the bottom section, I designed and developed the training process of two models: one that is responsible for imagining what the state of the agent would look like in the next step if it takes a specific action, and another model for determining which action should be taken in the next step to reach the target. I also integrated a curriculum learning process for both models to ensure that the agent starts learning from simple tasks first, before jumping to more difficult tasks. At the end of the process, the agent learned to plan and take the right sequence of actions to reach the target point on its own without any supervision or data collection, even when it was randomly initialized outside the training zone. The training process, the agent's behavior before and after the training, and the instructions for how to run the code are explained below. I also created the 3D simulation of the truck and environment and you can see the video below. In the video, you can see the performance of the controller model before the training process in ghost (red) trucks.


## Truck Kinematics

![Truck](https://raw.githubusercontent.com/ozyurtf/self-learning/main/figures/truck-kinematics.png)

## Training Emulator

![Truck](https://raw.githubusercontent.com/ozyurtf/self-learning/main/figures/emulator-training.png)

## Training Controller

![Truck](https://raw.githubusercontent.com/ozyurtf/self-learning/main/figures/controller-training.png)

## 3D Demo

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;border:0"
  src="https://www.youtube.com/embed/euWmj4p814I"
  title="YouTube video player"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  referrerpolicy="strict-origin-when-cross-origin"
  allowfullscreen></iframe>
</div>

## Trajectories

![Trajectory 1](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-1.png)

![Trajectory 2](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-2.png)

![Trajectory 2](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-3.png)

![Trajectory 4](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-4.png)

![Trajectory 5](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-5.png)

![Trajectory 6](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-6.png)

![Trajectory 7](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-7.png)

![Trajectory 8](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-8.png)

![Trajectory 9](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-9.png)

![Trajectory 10](https://raw.githubusercontent.com/ozyurtf/self-learning/main/trajectories/lesson-10-2025-06-04_02-09AM/trajectory-10.png)

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
    --test_y_cab_range -25 25
```

To train both emulator and controller models, run:

```bash
python truck-backer-upper.py --train_emulator True
```

To train only controller models, run:

```bash
python truck-backer-upper.py --train_controller True
```

## Run the 3D Simulation

The 3D interactive viewer generates a self-contained HTML file that opens in your browser. It replays the learned trajectories in a Three.js scene with a free camera, follow-cam mode, and a labeled training zone.

First, make sure the conda environment is active (see above). Then run:

```bash
python simulate_3d.py
```

This loads the trained controller (`models/controllers/controller_lesson_10.pth`), simulates 15 trajectories, writes `simulation_3d.html`, and opens it automatically.

**Controls inside the viewer:**

| Input | Action |
|---|---|
| Left-drag | Orbit camera |
| Right-drag | Pan camera |
| Scroll | Zoom |
| `↑ ↓ ← →` | Move camera forward / back / strafe |
| `Page Up / Page Down` | Move camera up / down |
| `Space` | Play / pause |
| `R` | Reset to first frame |
| Follow / Free buttons | Toggle follow-cam (front view of cab) |

To test with a custom spawn region:

```bash
python simulate_3d.py \
    --env_x_range 0 100 \
    --env_y_range -30 30 \
    --test_x_cab_range 40 90 \
    --test_y_cab_range -25 25
```

To change the number of trajectories displayed:

```bash
python simulate_3d.py --num_trajectories 10
```

# References

- Nguyen, D., & Widrow, B. (1989). *The truck backer-upper: an example of self-learning in neural networks*. In International 1989 Joint Conference on Neural Networks (pp. 357–363, vol. 2). [https://doi.org/10.1109/IJCNN.1989.118723](https://doi.org/10.1109/IJCNN.1989.118723)

- Schoenauer, M., & Ronald, E. (1994). *Neuro-genetic truck backer-upper controller*. In Proceedings of the First IEEE Conference on Evolutionary Computation. IEEE World Congress on Computational Intelligence (pp. 720–723, vol. 2). [https://doi.org/10.1109/ICEC.1994.349969](https://doi.org/10.1109/ICEC.1994.349969)
