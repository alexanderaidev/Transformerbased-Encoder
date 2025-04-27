The Algorithm itself works in "Cycles:

# 1: Maincycle
  Controls how often the overall training process is executed.

# 2: Progressive Cycle:
  Determines how many new files the model is trained on in each Main Cycle.  
  To avoid catastrophic forgetting, every time a new file is introduced, the model is retrained on `file(dataloader)[0]`.
  
# 3: Stabilization Cycle:
  The number of files in the stabilization set grows dynamically.  
  After each Main Cycle, the `dataloaders_stabilization` list is extended by up to N new files (e.g., 5).  
  You can configure the number of iterations per Stabilization Cycle.
  
   # What is an Iteration?
     One iteration means training on every file in the dataloader until the model reaches 91 % accuracy.  
     Typically, you should run 3–5 iterations per Stabilization Cycle.


** Note:**  
  Replace every occurrence of `Your path here` in the code with your own file or folder paths, then you’re all set.




**⚠️ Work in Progress:**  
- Stabilization-Dataloader currently on GPU will be directed to CPU
- Memory Leaks known, will be fixed at the next refactoring
