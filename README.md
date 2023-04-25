# planning
Diffusion Planning CS 282


Steps to run the code:

    1. run DatasetGenerator notebook
    1. run DiffusionTraining notebook



To do:
 - [ ] Make sure we haven't seen inference starting point,
   - I have already fixed seed for training, so this is just a double check. Plot the initial sampling starting condition (x,y),and the test set obs_0
   - np.random.seed() training and inference are disallined
 - [ ] Accuracy measured as number fineshed trajectories within n_step / total trajectories
 - [ ] Comment the load dataset functions
 - [ ] plug the diffusion in the loop
   - Option 1, actual train a diffusion to output the right acceleration, given the drone model
   - Option 2, train a diffusion to predict the next observation (we might use the model we already have)
     - For this option we are predicting the next position based on a fake model of drone. 
     - We use the 2D dataset and we let the drone follow the waypoint produced by the 8 action taken, 
and we produce the next 8 observation based on those. Those observations become the waypoint that the drone has to follow.
 - [ ] Fix dataset generator for drone, and re run experiment for drone only
   - Reduce the length of gdown decrease the parameters but doesn't increase the speed, same performances
   - Maybe introduce some plot for dataset generator,
   - Plot training loss and standard deviation along with hyperparameters.
