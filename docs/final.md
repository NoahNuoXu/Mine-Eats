---
layout: default
title: Final Report
---

## Video ##
<iframe width="560" height="315" src="https://www.youtube.com/embed/5Qh-5vsNqEA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary ##

Our Project is Serious-Sam, based on the game, has a arena of size 8 by 8 and has to kill a zombie within time limits. The goal of the project is to teach the agent, which we named Serious Sam, to kill the Zombie without damaging itself and the environment outside. We need to use AI/ML for this project because there are many strategies and features such as the location, angle, and surrounding area that affects the agent's performance, similar to when you are in a fighting club. Therefore, we decided to use deep q learning to achieve our goal for this project.

From the status, we have updated the size of our arena, the reward function, the number of Zombies and state space. For the final submission, we were able to improve upon the agent's performance. Before our agent was lucky a few times, but mainly was losing it's health while never even damaging the Zombie. We learned our mistake was the way we were calculating the observation space. It wasn't trivial to include the arena's wall location into the observation for the agent, because we assumed that the agent didn't need that information in order to kill the Zombie, however, later we learned that it is important because if the agent is facing the wall it cannot kill the Zombie. We also changed the reward function as we were learning that the reward function is very important to how the agent's performance is. A different reward function changed our agent's actions and events completely. We also learned that since killing a Zombie takes more than one hit, we needed to have the agent explore as much as it can to eventually kill the Zombie. For the final project, compared to the status project, the goal was to have the agent actually learn how to kill the zombie and keep improving with each step it takes.

Throughout the project, we had a lot of challenges and issues. One for example was the way to structure the reward function to optimize the time it takes for the agent and maximize the reward. To combat this problem, we had to try different reward functions and compare, which are listed below. Another issue we were having was that since the wall was able to be broken by the agent's sword the agent would sometimes break the wall and escape from the Zombie which was not our goal. Initially, we added the wall's location to our observation space which made the size of our observation much larger since the zombie's height was very large. However, when it would take random actions in the beginning it would break the wall a lot and didn't get any consequences, therefore we added a negative reward to any attacking of the wall. There were many more issues, however, with calculations and trial, we were able to get the agent to perform towards our goal.

<img src="https://cdn.discordapp.com/attachments/766872422996901921/790786866663522374/4.PNG"
     alt="Screenshot of Sam fighting Zombies"
     style="float: left; width:100%;" />
<!-- ![](https://cdn.discordapp.com/attachments/766872422996901921/790786866663522374/4.PNG) -->

<!-- ![](https://cdn.discordapp.com/attachments/766872422996901921/790786875874344970/5.PNG) -->

## Approach ##

We continued from Status with the deep q learning neural network approach for the agent to learn. We decided on deep q learning network because compared to using just q learning by itself, deep q learning will be able to take a much bigger observation and give you a result much faster. Since our observation was of size (9,5,5) due to the zombie's height, we needed to make sure that our model could handle a big observation. Though the equation we were using was the exact same to any q learning algorithm. We started with epsilon, our exploration rate, at 1, therefore the agent was exploring actions and states for majority of the beginning episodes. After that we had a epsilon decay of 0.99, so the agent start to learn by taking the actions that give the highest Q-value. Since initially the agent has no clue what action will lead to what result or reward, it was important to give the agent time to see the possibilities. However the main changes that we made were in the reward function. Since in the Status, the agent would keep dying and for each death, the agent received the same reward, it didn't learn to stay alive and try to kill the Zombie.

Some information on what our reinforcement learning model looked like will be listed below..

Deep Q Learning Network with two hidden layers, both of size 64.
State Size was (9,5,5) filled with 1's for the wall and 2's for the Zombie
Actions:

         Actions =  {
              0: 'move 1',  # Move one block forward
              1: 'turn 1',  # Turn 90 degrees to the right
              2: 'turn -1',  # Turn 90 degrees to the left
              3: 'attack 1'  # Destroy block
          }
 Learning Rate: 0.9
 Epsilon Decay: 0.99
 Max Steps: 10000





### Approach 1: Adding a positive timestep reward ###

First, we tried to do the opposite. We added a small positive reward for each time step the agent was alive, to encourage it to stay alive longer so it could have a higher chance of killing the Zombie. However this resulted in the opposite, instead the agent would learn to kill the Zombie at the end of the episode instead of the beginning since it was getting more reward from that. It also, sometimes would not kill the Zombie and be satisfied with the positive reward it accumulated just for being alive throughout the episode.


### Approach 2: Adding a negative timestep reward ###

Next, we tried to add a small negative reward for each step the agent was alive, in hopes to encourage the agent to kill the zombie faster. However, it did the opposite. The agent instead was more rushed to finish the episode and die quicker as that would lead to less negative reward.


### Approach 3: Removing the timestep reward ###

Finally, we decided to completely remove the timestep reward. This made the agent completely reliant on the zombies to increase or decrease his reward. This worked significantly better for training the agent because the agent wasn't pressured to finish the game in a certain way. On the contrast, it did lengthen the amount of episodes it took compared to the other approaches since the agent had to wait till the end of the episode to find out it's result. However, as mentioned in Approach 5, we needed to eventually add a penalty to attacking walls. We attempted to use the agent's sensing around him to prevent him from attacking walls. However, because the agent gets displaced by the zombies so frequently when being hit, he is unable to actually make accurate world state decisions. As such, adding a penalty to hitting the walls eventually taught the agent to stop hitting the walls.

Since there were only four cases that we could think of in which the agent needed reward: it hit a wall, it died, it killed a zombie, or the episode ended. We used the following code to manually add in the reward, since Malmo was struggling to register the reward.

        #add the reard for the agent hitting the wall
        if blocks:
            reward += (-500*blocks)

        #to ensure that if a zombie is killed we don't count any extra reward for zombies that are "off the screen"
        if episode_steps >= self.episodes:
            reward += 0

        #if the agent died it will lose points
        elif death:
            reward += -100
        # manually add the reward for killing a zombie
        else:
            reward += (1000**zombies_killed)

        return reward

### Approach 4: Editing the arena size and changing the observation###

Next, we tried changing around the arena size to give the agent more opportunity to learn. This worked really well. We ended up dropping the arena size down to 8x8, and this made the agent run into the zombies more.



On the inverse we also tried adjusting the arena size to be nearly infinite, but the agent would have such low success actually attacking the zombies that it would learn to pretty much just run. This didn't work particularly well.

To add the zombie's location to our observation state we used the following code that would find all entities that were a zombie. After this we had to rotate the matrix to the direction that the agent was facing to make it in it's perspective.

                   for e in entities:

                        if e[u'name'] == 'Zombie':
                            zombies_count += 1
                            x = int(e['x'])
                            z = int(e['z'])
                            y = e['y']

                            if abs(x-xpos) <= halfway and abs(z-zpos) <= halfway:
                                 i = x - xpos + halfway
                                 j = z - zpos + halfway


                                 #had to flip i and j to match row and column to the x,z coords in malmo
                                 obs[int(y)+1][int(j)][int(i)] = 2

 We also used the Malmo platform to gain information of the surrounding information of the agent to incorporate the wall's location. This helped as it gave the agent referrence as to where it is so it knew more when it was calculating the next Q value.


### Approach 5: Making the initial attack smarter ###

Next, we tried to add a small radar system that allows the agent to detect zombies within a certain radius and attack if they are in that radius. The problem here was that the zombies would have a propensity to attack from behind. This caused the agent to get stuck attack at nothing but air, and he would learn that attacking is actually NOT what he wants to do, as the zombies would eventually push the agent against a wall, and as he attacks a wall, the negative reward would accumulate.

        attack_binary = 0;
        SamPos = [radar[0]['x'],radar[0]['z']]
        for x in radar:
            if x['name'] == 'Zombie':
                zombiePos = [x['x'],x['z']]
                distancer = math.sqrt(((SamPos[0]-zombiePos[0])**2)+((SamPos[1]-zombiePos[1])**2))
                print("distancer ",distancer)
                if distancer < 10:
                    if zombiePos[0] < SamPos[0]
                        attack_binary = [1, 1];
                    else:
                        attack_binary = [1, 0];
                    break;


### Conclusion ###
 We decided finally on using a mix of approach 3,4,and 5. We removed the reward for time steps because it was causing the agent to go in the wrong direction as we intended. We added the extra information to the observation space because we realized later on that it would be really important information for the agent to know this, otherwise it is going to have difficulty decided the next action.


## Evaluation ##

### Quantatitive ###
<img src="https://media.discordapp.net/attachments/766872422996901921/790778326070329414/average_returns.png"
     alt="Zombie Killer graph"
     style="float: left; margin-right:100%;" />
<!-- ![](https://media.discordapp.net/attachments/766872422996901921/790778326070329414/average_returns.png) -->
Here, we can see a graph of 10000 steps. The agent learns very quickly at first, but seems to fall into a plateau before fixing its behavior. This could be due to the fact that the rewards are limited to only the zombie interaction. If the player dies, they get penalized. If the player hits a zombie, they get rewarded. It's fairly simple to think about, but because all the actions are randomized at first, until the agent starts building a dataset of when it hits the zombie, (which is very unlikely near the beginning) the learning rate is very slow after the initial spurt.


<img src="https://cdn.discordapp.com/attachments/766872422996901921/790663342637318144/returns.png"
     alt="Zombie Killer graph"
     style="float: left; margin-right:100%;" />
If we zoom in, and only do a test of 1600 steps, we can see a beautiful and ideal learning graph
<!-- ![](https://cdn.discordapp.com/attachments/766872422996901921/790663342637318144/returns.png) -->
But there's something here that's rather interesting. In the graph of 10000 steps, the agent was unable to learn fast enough to have results over 0 until the end. However, in the graph of 1600 steps, we seemed to have significantly better end-performance. This is interesting to think about. Is the agent perhaps being given too many steps? Regardless, we were able to finally find a way to make our agent learn how to get scores above 0, and begin actively attacking and killing zombies. We think that intially since majority of the agent's actions are random and not the max argument it was able to kill the zombie almost by luck, which then it used to learn later on, but was still struggling. Since the observation matrix was of size (9,5,5) which can have many possibilities as to where the zombie and wall is in perspective to the agent, we probably needed to do more way steps so the agent can explore all of the states in the state space. However, since this took about three hours to run, it became difficult to run more steps. However, you can see the pattern of the agent's learning in the graph. It would explore and then find out something that helps it, then it keeps doing that until it finds something else to improve it, which then it exploits,and this pattern would continue.


### Qualitative ###

<img src="https://cdn.discordapp.com/attachments/766872422996901921/790786875874344970/5.PNG"
     alt="Screenshot of Sam fighting Zombies"
     style="float: left; width:100%;" />
The agent would, as seen on the graph, initially either die or kill the zombie, but after some time it was getting better at killing the Zombie. It look a very long time for the agent as seen in the first plot to improve, however after it improved then it would stay stable until it was able to explore more. It was a balance of exploration and exploitation to ensure that the agent was doing the right thing. Our goal at the beginning of the project was to have the agent learn to kill the zombie in front of it and it did!


## Conclusion ##


This project taught us a lot about the combination of using deep learning with reinforcement learning. We learned that in machine learning there are more issues than just trying to train the model, but providing the agent a safe and optimal environment to thrive. We were glad to get the agent to kill the zombie after a lot of hard work. There are still some areas we could improve on, such as reducing the state space, because as of right now it is very big or even running more episodes allowing the agent to explore more. The next challenge would be to include more zombies and challenge the agent to work hard.



## References ##
build_test.py, mob_fun.py, tabular_q_learning.py and moving_target_test.py from Canvas page. <br />
https://eg.bucknell.edu/~cld028/courses/379-FA19/MalmoDocs/classmalmo_1_1_mission_spec-members.html <br />
https://microsoft.github.io/malmo/0.17.0/Python_Examples/Tutorial.pdf <br />
https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/ <br />
https://github.com/microsoft/malmo/blob/master/Schemas/Types.xsd <br />
https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html#element_Weather <br />
https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
