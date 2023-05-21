# AI Method 2 - Q-Learning

We utilised Q-learning as a reinforcement learning mechanism to continually improve our agent's action selection. Given the time limit, offline planning becomes a necessity, as we cannot afford the time to explore every possible action during gameplay.


# Table of Contents
- [AI Method 2 - Q-Learning](#ai-method-2---q-learning)
- [Table of Contents](#table-of-contents)
  - [Governing Strategy Tree](#governing-strategy-tree)
    - [Motivation](#motivation)
    - [Application](#application)
    - [Solved Challenges](#solved-challenges)
    - [Trade-offs](#trade-offs)
      - [*Advantages*](#advantages)
      - [*Disadvantages*](#disadvantages)
    - [Future improvements](#future-improvements)

## Governing Strategy Tree  

### Motivation  


[Back to top](#table-of-contents)

### Application  
Q-learning is highly applicable in the context of Azul gameplay. By applying Q-learning, we aim to develop an AI agent that can play the game autonomously and efficiently. The agent learns optimal strategies over time by interacting with the game environment, understanding different states, and making decisions based on its past experiences. It learns to choose actions that maximize the expected cumulative rewards, adjusting its strategy as it gains more experience. This includes strategic tile selection, optimal tile placement, managing penalties, and counter-play to minimize opponent's score. This capacity for adaptive, strategic gameplay makes Q-learning a valuable tool for developing a competitive AI player for Azul.


[Back to top](#table-of-contents)

### Solved Challenges
For classic Q-learning approach, which uses a Q-table to store values for each state-action pair, becomes unfeasible due to space complexity when the state-action space is too large, as it is in many real-world problems. 

In order to solve this problem, approximate Q function was implemented. It represents Q(s,a) as a linear combination of features and their weights. This approach enables generalisation across states and actions, reducing the amount of data we need to store and compute, making the learning process significantly more efficient and feasible.
[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  


#### *Disadvantages*

[Back to top](#table-of-contents)

### Future improvements  

[Back to top](#table-of-contents)
