# Truck Backer Upper 

## Updates 

- There is no manual process in the notebook anymore.
  - Lessons are created automatically instead of manually. All we need to do is specify the first lesson configuration, the last lesson configuration, and the number of lessons we want to use.
  - Previously, I was specifying the maximum number of steps the truck can take during the training of the controller for each lesson manually. Now, I use a fixed number for all lessons, and it seems to be working well.
- The reset() function in the Truck class has been updated to make the code cleaner.
- A new update_state() function has been created in the Truck class. The state of the truck is now updated with this function during the training of the controller. This improved the code's clarity.
- Previously, optimizer.zero_grad() was used after optimizer.step(). It is now placed before loss.backward().
- I added visualizations of gradients and loss values to Wandb instead of tracking them in Tensorboard and/or notebook.
- I visualized the computational graph for both the controller and training process. The visualization of computatinal graph for the controller can be seen in computational-graphs folder and the visualization of the training process can be found in figures folder.
