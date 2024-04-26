The purpose of this document is to summarize the findings of the finetuning team from Winter 2024.

The mmvae/finetuners module contains a trainer that takes in a csv file that has the knn correlations from diseased to healthy cells and trains a single modal vae on homo sapian genetics.
The proccess can be outlined as follows: 
   - Train a model to reconstruct healthy cells
   - Compute KNN on dataset to find correlated diseased to healthy cells
   - Sample the 100 most similar healthy cells for each diseased cell reference to form dataset
   - Load pre-trained model and begin training on diseased cells where healthy cell passed in and diseased cell used as reference for computing loss.

 We explored two approaches to make this happen:

  The first approach made no modifications to the pre-trained model and updated all of the model parameters while training.

  The second approach involved freezing all of the parameters in the pre-trained model and adding a layer after the resampling layer of the VAE.
  This "finetune" layer was the only parameters that were updated while training on diseased cells.

The important take aways from this work is how the reconstruction of the diseased cells affects the generalizablity of the model.
When updating only the finetune layer parameters we have shown that the KL divergence does not get affected. This is advantages as 
we can then turn off the finetune layer and still use the model for predicting healthy cells. 

Another important take away is the balance of the weight of KL divergence loss. From experimentation we have derived that annealing KL with a warm start to a beta of
roughly 0.15 proves to be optimal to balance the reconstruction with the kl divergence however this can be affected by the model architure and training so it will remain
an ongoing investigation as architectures progress.

Overall, the finetuners module contains the nessary fuctions to reconstruct the expierements described above. We have shown that it is possible to predict diseased cells from
healthy references but need to quantify the generalizablity of the model.
