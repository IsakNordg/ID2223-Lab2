# ID2223-Lab2

## Task
Task 2: Improve pipeline scalability and model performance
1. Describe in your README.md program ways in which you can improve
model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the
fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to
train a better model that one provided in the blog post
If you can show results of improvement, then you get the top grade.
2. Try out fine-tuning a couple of different open-source foundation LLMs to
get one that works best with your UI for inference (inference will be on
CPUs, so big models will be slow).


## Idea of the project
My idea for fine-tuning the model into being good at performing a specific task was to use it as a math problem solver, somewhat like Wolfram Alpha! For this, i used a dataset consisting of mathematical questions and their answers, and fine-tuned the model on this dataset: https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT

I also did some hyperparameter tuning to improve the model performance.

## Ways to improve model performance
### Model-centric approach
**Hyperparameter tuning**: Hyperparameters are parameters that are not learned during training. They are set before training and control the learning process, and thus how the model will be optimized. The hyperparameters that I have taken into consideration are:

*r*: Rank of the LoRA decomposition. This was chosen to be 16, as it was the best performing rank in the original blog post.

*lora_alpha*: The scaling factor for how much the LoRA matrix should influence the final prediction. This was chosen to be 16 as well, as it was the best performing value in the original blog post.

*lora_dropout*: The dropout rate for the LoRA layer. This was chosen to be 0.05, which is pretty low but not zero, as i wanted to avoid overfitting but not impact training time too much, as resources were limited during training.

*Packing*: Packing is used to add short sequences together in the input, increasing speed.

*Weight decay*: I used a weight decay of 0.01 to prevent overfitting by penalizing large weights in the model. This helps the model generalize better to unseen data, as it discourages the model from excessively fitting to the training dataset.

*Learning rate scheduler*: I used a linear learning rate scheduler because it gradually reduces the learning rate over time, ensuring that the model converges smoothly and avoids overshooting the optimal solution. This is especially useful when fine-tuning large models, as it allows for stable updates to the weights during training.

### Data-centric approach
In order to achieve the desired functionality of a math problem solver, I used a dataset consisting of mathematical questions and their answers, and fine-tuned the model on this dataset: https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT

The dataset consists of 100,000 math problems and their solutions, and was chosen because it only outputted the result of the math problem, and not the steps to solve it. This way, the model would generate fast answers, meaning you don't have to wait too long for the correct answer, given that we are running the model on a CPU, which is slower than a GPU.

## Results
The model was able to generate answers to math problems, but the answers were often incorrect when the problems get more complex. It is able to answer things like 2+2=4, sqrt(123), 75*9, etc, but struggles with e.g. solving 2x+15 = 25 for x and similar. This means that it can be used as a simple calulator, but not as a math problem solver, which makes it not very useful as there are real caluclators which can solve these kinds of direct problems.

This low performance is likely due to the fact that the model was not trained on the steps to solve the math problems, but only the answers. This means that the model does not know how to solve the problems, but only what the answer is if it has been seen in the data. As said, this is not very useful.

## Fine-tuning different open-source foundation LLMs
I also tried fine-tuning a larger model (3B parameters instead of 1B), but the preliminary results, after just a few round of traning I tried the model and decided that its answers were too slow for a CPU to be useful, and the answers were not significantly better than the smaller model. It is of course hard to decide how much it improved after just a few rounds of training, but this proxy was enough for me to decide that the larger model was not worth the extra time and resources.

