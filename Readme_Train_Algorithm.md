The Algorithm itself works in "Cycles:

# 1: Maincycle
  It controlls how often the algorithm is used.

# 2: Progressive Cycle:
  Controlls how many new file the AI is being trained on.
  With every new file, the model is trained on file(dataloader)[0] again, to avoid Catastrophic Forgetting.

# 3: Stabilization Cycle:
  The amount of files used is dynamicly growing. For every main cycle, the "dataloaders_stabilization"-list is growing for the same amount as the progressive circle handles at max (e.g. 5).
  The Stabilization Cycle can handle different amount of iteration.
   # What is an Iteration?
     One single Iteration means that every file inside of the dataloader is trained on an accuracy of 91%. 
     Usually you should go for 5 Iterations per Stabilization Cycle.
