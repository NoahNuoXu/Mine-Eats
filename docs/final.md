---
layout: default
title: Final Report
---

## Video ##
<iframe width="560" height="315" src="https://www.youtube.com/embed/5Qh-5vsNqEA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary ##

Our Project is Serious-Sam, based on the game, has a arena of size 8 by 8 and has to kill a zombie within time limits. The goal of the project is to teach the agent, which we named Serious Sam, to kill the Zombie without damaging itself and the evnironment outside. We need to use AI/ML for this project because there are many strategies and features such as the location, angle, and surrounding area that affects the agent's performance, similar to when you are in a fighting club. Therefore, we decided to use deep q learning to achieve our goal for this project.

From the status, we have updated the size of our arena, the reward function, the number of Zombies and state space. For the final submission, we were able to improve upon the agent's performance. Before our agent was lucky a few times, but mainly was losing it's health while never even damaging the Zombie. We learned our mistake was the way we were calculating the observation space. It wasn't trivial to include the arena's wall location into the observation for the agent, because we assumed that the agent didn't need that information in order to kill the Zombie, however, later we learned that it is important because if the agent is facing the wall it cannot kill the Zombie. We also changed the reward function as we were learning that the reward functrion is very important to how the agent's performance is. A different reward function changed our agent's actions and events completely. We also learned that since killing a Zombie takes more than one hit, we needed to have the agent explore as much as it can to eventually kill the Zombie. For the final project, compared to the status project, the goal was to have the agent actually learn how to kill the zombie and keep improving with each step it takes.
<img src="https://cdn.discordapp.com/attachments/766872422996901921/790786866663522374/4.PNG"
  alt="Screenshot of Sam fighting Zombies"
  style="float: left; width:100%;" />
<img src="https://cdn.discordapp.com/attachments/766872422996901921/790786866663522374/4.PNG"
  alt="Screenshot of Sam fighting Zombies"
  style="float: left; width:100%;" />
<!-- ![](https://cdn.discordapp.com/attachments/766872422996901921/790786866663522374/4.PNG)
![](https://cdn.discordapp.com/attachments/766872422996901921/790786875874344970/5.PNG) -->

## Approach ##

We continued from Status with the deep q learning neural network approach for the agent to learn. We decided on deep q learning network because compared to using just q learning by itself, deep q learning will be able to take a much bigger observation and give you a result much faster. Since our observation was of size (9,5,5) due to the zombie's height, we needed to make sure that our model could handle a big observation. Though the equation we were using was the exact same to any q learning algorithm. We started with epsilon, our exploration rate, at 1, therefore the agent was exploring actions and states for majority of the beginning episodes. After that we had a epsilon decay of 0.99, so the agent start to learn by taking the actions that give the highest Q-value. Since initially the agent has no clue what action will lead to what result or reward, it was important to give the agent time to see the possibilities. However the main changes that we made were in the reward function. Since in the Status, the agent would keep dying and for each death, the agent received the same reward, it didn't learn to stay alive and try to kill the Zombie.


### Approach 1: Adding a positive timestep reward ###

First, we tried to do the opposite. We added a small positive reward for each time step the agent was alive, to encourage it to stay alive longer so it could have a higher chance of killing the Zombie. However this resulted in the opposite, instead the agent would learn to kill the Zombie at the end of the episode instead of the beginning since it was getting more reward from that. It also, sometimes would not kill the Zombie and be satisfied with the positive reward it accumulated just for being alive throughout the episode.


### Approach 2: Adding a negative timestep reward ###

Next, we tried to add a small negative reward for each step the agent was alive, in hopes to encourage the agent to kill the zombie faster. However, it did the opposite. The agent instead was more rushed to finish the episode and die quicker as that would lead to less negative reward.


### Approach 3: Removing the timestep reward ###

Finally, we decided to completely remove the timestep reward. This made the agent completely reliant on the zombies to increase or decrease his reward. This worked significantly better for training the agent. However, as mentioned in Approach 5, we needed to eventually add a penalty to attacking walls. We attempted to use the agent's sensing around him to prevent him from attacking walls. However, because the agent gets displaced by the zombies so frequently when being hit, he is unable to actually make accurate world state decisions. As such, adding a penalty to hitting the walls eventually taught the agent to stop hitting the walls.

### Approach 4: Editing the arena size ###

Next, we tried changing around the arena size to give the agent more opportunity to learn. This worked really well. We ended up dropping the arena size down to 8x8, and this made the agent run into the zombies more.



On the inverse we also tried adjusting the arena size to be nearly infinite, but the agent would have such low success actually attacking the zombies that it would learn to pretty much just run. This didn't work particularly well.


### Approach 5: Making the initial attack smarter ###

Next, we tried to add a small radar system that allows the agent to detect zombies within a certain radius and attack if they are in that radius. The problem here was that the zombies would have a propensity to attack from behind. This caused the agent to get stuck attack at nothing but air, and he would learn that attacking is actually NOT what he wants to do, as the zombies would eventually push the agent against a wall, and as he attacks a wall, the negative reward would accumulate.



## Evaluation ##

<img src="https://media.discordapp.net/attachments/766872422996901921/790778326070329414/average_returns.png"
  alt="Zombie Killer graph 10000 Steps"
  style="float: left; margin-right:100%;" />
<!-- ![](https://media.discordapp.net/attachments/766872422996901921/790778326070329414/average_returns.png) -->
Here, we can see a graph of 10000 steps. The agent learns very quickly at first, but seems to fall into a plateau before fixing its behavior. This could be due to the fact that the rewards are limited to only the zombie interaction. If the player dies, they get penalized. If the player hits a zombie, they get rewarded. It's fairly simple to think about, but because all the actions are randomized at first, until the agent starts building a dataset of when it hits the zombie, (which is very unlikely near the beginning) the learning rate is very slow after the initial spurt.


<img src="https://cdn.discordapp.com/attachments/766872422996901921/790663342637318144/returns.png"
  alt="Zombie Killer graph 1600 Steps"
  style="float: left; margin-right:100%;" />
If we zoom in, and only do a test of 1600 steps, we can see a beautiful and ideal learning graph
<!-- ![](https://cdn.discordapp.com/attachments/766872422996901921/790663342637318144/returns.png) -->
But there's something here that's rather interesting. In the graph of 10000 steps, the agent was unable to learn fast enough to have results over 0 until the end. However, in the graph of 1600 steps, we seemed to have significantly better end-performance. This is interesting to think about. Is the agent perhaps being given too many steps? Regardless, we were able to finally find a way to make our agent learn how to get scores above 0, and begin actively attacking and killing zombies.



## References ##
build_test.py, mob_fun.py, tabular_q_learning.py and moving_target_test.py from Canvas page. <br />
https://eg.bucknell.edu/~cld028/courses/379-FA19/MalmoDocs/classmalmo_1_1_mission_spec-members.html <br />
https://microsoft.github.io/malmo/0.17.0/Python_Examples/Tutorial.pdf <br />
https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/ <br />
https://github.com/microsoft/malmo/blob/master/Schemas/Types.xsd <br />
https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html#element_Weather <br />
https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
