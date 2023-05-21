# AI Method 2 - Q-Learning

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

- Complexity:
  * Azul is a game with a large state space and many possible actions. This, combined with the element of chance introduced by the random selection of tiles, makes it a challenging task for traditional algorithms. Q-learning, with its ability to handle large state spaces and learn from exploration, becomes a suitable choice.
  
- Real-Time Decision Making: 
  * The agent needs to make decisions within a certain time limit in each turn. Q-learning's ability to update Q-values iteratively and use the current Q-values for decision-making fits this requirement well. Which maight result a better outcome than other online planning method.

[Back to top](#table-of-contents)

### Application  
Q-learning is highly applicable in the context of Azul gameplay. By applying Q-learning, we aim to develop an AI agent that can play the game autonomously and efficiently. The agent learns optimal strategies over time by interacting with the game environment, understanding different states, and making decisions based on its past experiences. It learns to choose actions that maximise the expected cumulative rewards, adjusting its strategy as it gains more experience. This includes strategic tile selection, optimal tile placement and managing penalties. This capacity for adaptive, strategic gameplay makes Q-learning a valuable tool for developing a competitive AI player for Azul.


[Back to top](#table-of-contents)

### Solved Challenges
For classic Q-learning approach, which uses a Q-table to store values for each state-action pair, becomes unfeasible due to space complexity when the state-action space is too large, as it is in many real-world problems. 

In order to solve this problem, approximate Q function was implemented. It represents Q(s,a) as a linear combination of features and their weights. This approach enables generalisation across states and actions, reducing the amount of data we need to store and compute, making the learning process significantly more efficient and feasible.
[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  

- Offline Learning:
  * The agent learns after the game is over, which allows for more thoughtful analysis and long-term strategy development without being restricted by a time limit.
  
- Adaptability:
  * The Q-learning algorithm can adapt its policy over time based on its experiences, allowing it to potentially improve its performance as it plays more games.
  
- Manageable State Space:
  * By using a linear approximation function and features instead of a Q-table, the approach effectively manages the large state space of the Azul game, reducing memory and computational requirements.

#### *Disadvantages*
- No optimal Solution:
  * In games like Azul that have a significant element of randomness, developing an optimal strategy that works perfectly for every game is indeed a very challenging or even impossible.
  
  * It is only possible to achieve a 'good' strategy, since the distribution of tiles to the factories is randomised. Even if we could learn an optimal strategy given a certain game state, the opponent's actions could always take the game in an unforeseen direction. 
  
- Approximation Accuracy
  * Using a linear approximation function to represent the Q-function might not capture the full complexity of the Azul game, potentially leading to suboptimal play.

[Back to top](#table-of-contents)

### Future improvements  
- Feature Representation:
  * The performance of Q-learning heavily depends on the features used to approximate the Q-function. Hence, feature engineering can be the area to fucus on. 
  
- Multi-step Learning:
  * This approach takes into account multiple future steps when updating the Q-values instead of just the immediate reward, which could lead to more far-sighted learning. Methods such as n-step Q-learning or Monte Carlo Tree Search can be investigated in this direction.
  
[Back to top](#table-of-contents)
