# AI Usage Disclosure

**Tools used**: ChatGPT, Codex and Junie

## **What I used AI for**
I used AI for many things in this project. Mainly to speed up my workflow. AI wrote most of the new model 
implementations in `models.py` I directed it what to do and it did it for me and I would check afterward to make 
sure that it was correct and did what I wanted. I also had AI write some of my experiments in the notebooks as these followed
a very similar pattern and was easy for it to replicate. I would have it create sweeps for me and then I would go in and update
the parameters to better values. AI also created the new RidgeRegressionTrainer to train the ESNs. AI also wrote a lot of the new 
helper functions for creating the stock datasets like `build_stock_feature_df`. Also the `README.md` was written by AI mostly with credits added by me.
Also the beginning of `ESN Exploration.ipynb` was written by AI and it did some weird stuff that I did not ask it to do.

## **What was my own**
The big thing that was my own in this code base was the ideas behind and the design of the hybrid ESN GRU model. This is an idea
that I came up with and then AI helped me expand upon and implement it. Also in `financial_data.py` the `make_single_stock_df` was completely created by me.
Also a majority of the features used in the function were my idea and a few were from AI. It was also my idea to create a new Trainer for ridge regression that 
still worked with the original TrainerConfig and syntax. Also the new additions to the ModelConfig were all my idea and then implemented by AI.
Most of the core ideas and design were from me and then AI implemented them to speed up the process and created helped functions in some cases that were not my idea.

## **Checking AI output**
How I would normally check AI output is by going to the files that it modified and reading over the code. While reading the code
I would try and understand what it was doing and if it was the same as what I had planned. If it was not and it was a small fix I would do it
myself but if it was a big change I would describe in detail what I would like changed. I some things wrong like when I originally
asked it to implement my idea of the hybrid ESN GRU it did it in a different way then I asked only adding in the ESN state at the very end not 
at each time step. So I prompted it in more detail to correct this error. Also when having it create the sweeps it would but in some strange values
that I would need to change. In `ESN Exploration.ipynb` if you look at the beginning of the notebook that was completely AI written I did not ask it to do
that much I just wanted it to run a short test and create a sweep. However this output was interesting to look at and provided some value.

## **What I learned**
I learned a lot this is the most extensively I have every used AI to write code and the most I have ever used AI code agents before.
What I learned is that AI is very good at writing large amounts of code however you need to be very specific and spend time 
crafting your prompt because if you leave any ambiguity it may do things that you did not intend or the wrong way. What this
taught me is that I need to be specific in my prompts and give it as much context as possible and give it a detailed design of 
what you want it to do. This is something that I will be more carefull with in the future because there were numerous times where
I wasn't specific and it did things I did not ask for. It also remined me of good SWE practices like creating helper functions.
This also showed me that I should more throughly plan before prompting my AI so that I give it the design that I want and think
is best for the code base. It also reinforced how good AI is at writting documentation.