# Truck Backer Upper 

## Updates 

- There is no manual process in the notebook anymore.
  - Lessons are now created automatically instead of manually. All we need to do is specify the first lesson configuration, the last lesson configuration, and the number of lessons we want to use between these two lessons.
  - Previously, I was specifying the maximum number of steps the truck is allowed to take during the training of the controller manually. And I was doing this for each lesson separately. Now, I use a fixed number for all lessons, and it seems like it is working well.
- The reset() function in the Truck class is updated to make the code cleaner.
- A new update_state() function has been created in the Truck class. The state of the truck is now updated with this function during the training of the controller. This improved the code's clarity.
- I added visualizations of gradients and loss values to Wandb instead of tracking them in Tensorboard and in the notebook. They can be seen in the links below:
  - [Emulator Training](https://api.wandb.ai/links/furkanozyurt21/ciflisl6)
  - [Controller Training](https://api.wandb.ai/links/furkanozyurt21/hgxga7y0)
  
- I visualized the computational graph for both the controller and training process. The visualization of computatinal graph for the controller can be seen in computational-graphs folder and the visualization of the training process can be found in figures folder.
