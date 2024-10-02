Demonstrate search predict


DSPY - Prompt programming and not prompt engineering

Generates an optimised prompt for a given dataset. 

It is a framework for Algorithmically optimizing LM Prompts and weights.

Foundationl Units
  - Language Models
  - Signatures
  - Modules
  - Data
  - Metrics
  - Optimizers
  - Assert and Suggest

  Signature - Can be considered as definition for a prompt or metadata for the prompt. 
              This defines the input and output for LMs
  Modules - It is called as DSpy program 

  Predictors - Is what calls the LLM using our signature and it know how to leverage our signature. 
  TypedPredictors
  TypedChainOfThought

  Module - Wraps the Predictors

  Evaluate 

  Task 8 - Progra Optimization - Teleprompting
   Take the program, a training set and a metric and makes changes/tweals to our program to improve our metrices on our dataset.


Another Optimizer - BootstrapFewShot 

Multi Hop QA Module
  Generate signature and module
  