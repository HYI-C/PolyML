# Notes
- The continuous features are currently now each entry in the vertex matrix. The
  loss function we should employ is thus the normal MAE loss (so the shape gets
  as close as possible). Is there an invariance I should be aware of (and take
  care of?)
# To Do: High priority 
1. Implement the MAE loss
2. We still need to fix up the offsets bit in encoder.py
3. Fix up the data plugins
4. Fix the target creation, try with real data.

# To Do: Longer term priority
1. To pad the number of vertices, we should have a script that generates
   "vertices" within a single line, or slightly perturbed.
2. The data input can either be of the form [[vertex1], [vertex2], etc.] or n-
   datasets (of n vertices) of the form [shape1, shape2, shape3, etc.], [shape1,
   shape2, shape3,...]