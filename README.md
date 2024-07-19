## Truck Backer Upper

### Situation and Task
In this project, there was a simulation of a truck and an environment. It could only be able to move backwards. It's moving speed was fixed. Therefore, the only variable was the steering angle. Every time we specify the steering angle, the truck was moving one step back based on this stering angle. 

The truck was initialized to different positions randomly in the environment. And the goal was to back up the truck to the target position properly no matter where it is initialized. But there were some challenges. 

For instance, let's say that the truck is initialized to a position randomly. In that position, we can basically use infinite number of steering angles between 0 and 360. And the new position of the truck will be different with each one of these infinite number of steering angles. After we pick one of them and the truck move backwards according to this angle, it will have infinite number of routes that it can follow to reach the target position again. So each step basically creates infinitely large solution space. This was one of the first problems. 

One note is that, the degree between the head of the truck and trailer should not be more than 90 degree at anytime. Otherwise this would be against the law of physics. So another issue was to make sure that this condition is met from the beginning and end of the solution. 

Also, even if the truck learns how to back up from a specific location, it has to generalize this knowledge across all kinds of different locations and positions. 

And similarly, there are many different routes to follow from one location to back up the truck. The should be to find the shortest possible path. 

Other big problem was that there wasn't any data that I could use to make the truck aware of the consequences of its actions. 

So, these were the some of the main challenges that needed to be solved in the project. 

####  Simulation Before Training
![Truck Backer Upper GIF](videos/truck-backer-upper-before.gif)

### Action

If we look at human or any other animals' perspective, we can be able to do vast majority of things such as going from one location to another, sitting into somewhere, moving our arms and hands to eat food because the internal representation of the world in our brains is aware of the consqeuences of its actions. And we learned it by ourselves just performing the actions and seeing the results. 

I prepared a similar system for this project by creating a list of random stering angles, and a feed-forward neural network model, and teaching this model the effects of moving backwards with these steering angles. This model is called emulator. At the end of the training process, it was predicting the new position of the truck with all kinds of different steering angles very accurately. In other words, it was aware of the consequences of the truck's actions before those actions are taken. 

After preparing the model that is aware of the consequences of the actions of the truck in various different locations, the next step was to solve the other problems. 

In the next step, I prepared another model that is similar to recurrent neural network. When the truck is initialized to a random position and makes a couple of steps, the next time it chooses the steering angle, it takes all the actions that are made until that time when selecting the steering angle. 

To make things easier for the model, I started the training process from the positions that are very close to the target position. Then

### Results

#### Simulation After Training
![Truck Backer Upper GIF](videos/truck-backer-upper-after.gif)
