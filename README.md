# Alpha Zero General (any game, any framework!)

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch, Keras, TensorFlow and Chainer. An accompanying tutorial can be found [here](http://web.stanford.edu/~surag/posts/alphazero.html). We also have implementations for GoBang and TicTacToe.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

To start training a model for Othello:
```bash
python main.py
```
Choose your framework and game in ```main.py```.

### Docker Installation
For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:
```
./setup_env.sh
```
to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:
```
docker exec -ti pytorch_notebook python main.py
```

### Experiments
We trained a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn). This took about 3 days on an NVIDIA Tesla K80. The pretrained model (PyTorch) can be found in ```pretrained_models/othello/pytorch/```. You can play a game against it using ```pit.py```. Below is the performance of the model against a random and a greedy baseline with the number of iterations.
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

A concise description of our algorithm can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf).

### Contributing
While the current code is fairly functional, we could benefit from the following contributions:
* Game logic files for more games that follow the specifications in ```Game.py```, along with their neural networks
* Neural networks in other frameworks
* Pre-trained models for different game configurations
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 
* Asynchronous MCTS as described in the paper

### Contributors and Credits
* [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.
* [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
* [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
* [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
* [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.
* [Adam Lawson](https://github.com/goshawk22) contributed rules and a trained model for 3D TicTacToe.





```shell




[PITTING] random vs random

   0 1 2 3 4 5 
-----------------------
0 |X O O O O X |
1 |X O O O X X |
2 |X X O O X X |
3 |X X O O X X |
4 |X O X X O X |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O O X X |
2 |X X X O O X |
3 |X X X X O X |
4 |X X O O X X |
5 |X X O O X X |
-----------------------
(2, 0, 0)




[PITTING] random vs greedy
   0 1 2 3 4 5 
-----------------------
0 |X X O O O O |
1 |X X X O O X |
2 |X X O X O X |
3 |X X O O X X |
4 |X X O X X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |O X O O X O |
2 |O X O X X O |
3 |O X X X X O |
4 |O X X O O O |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] random vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X - |
2 |X X X X X O |
3 |X X X X X - |
4 |X X X X X X |
5 |X X X X X - |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X O O O O |
1 |O X O O O O |
2 |O O X O O O |
3 |X X O O O O |
4 |O O O O O O |
5 |O O O O - O |
-----------------------
(0, 2, 0)




[PITTING] random vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |O - X X X X |
1 |O X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X X X X O X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X X O O |
2 |O X O O X O |
3 |O O X X O O |
4 |O X O X X O |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] random vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |O X X X O X |
2 |O O X X O X |
3 |X X O X O X |
4 |X X O O X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X X O |
2 |O X O X O O |
3 |O O O O O O |
4 |O O O O O X |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] random vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X O X X X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O O |
2 |O O X X O X |
3 |O O X O X X |
4 |X O O X X X |
5 |X O O O O O |
-----------------------
(0, 2, 0)




[PITTING] greedy vs random
   0 1 2 3 4 5 
-----------------------
0 |X X X X O X |
1 |X X X O O O |
2 |X X O X O O |
3 |X X O X O O |
4 |X X X O O O |
5 |X X O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X O O |
2 |X X X O X O |
3 |X X O X X O |
4 |X X X O X O |
5 |X X X X X X |
-----------------------
(1, 1, 0)




[PITTING] greedy vs greedy
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X X X O O |
3 |X X X O O O |
4 |X X O X O O |
5 |X O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X X X O O |
3 |X X X O O O |
4 |X X O X O O |
5 |X O O O O O |
-----------------------
(1, 1, 0)




[PITTING] greedy vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X X O |
2 |X X X X X O |
3 |X X X X X O |
4 |X X X X X O |
5 |X X X X - O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X O O |
2 |O O X O O O |
3 |O O O O X O |
4 |O O O O O O |
5 |- O O O O O |
-----------------------
(0, 2, 0)




[PITTING] greedy vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |X X X X - X |
1 |X X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X - X X - X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O X X |
1 |O O O O O - |
2 |O O O O O O |
3 |O O X O O O |
4 |O O O O X O |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] greedy vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O O X X X |
3 |X O O X X X |
4 |X O O O X X |
5 |X X X X O X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
(0, 2, 0)




[PITTING] greedy vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
(0, 2, 0)




[PITTING] alphabeta-simple vs random
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O - |
2 |O O O O - O |
3 |- O O O O O |
4 |O O O O O O |
5 |- - - O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X O X X X X |
1 |X X O X O X |
2 |X X X O O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] alphabeta-simple vs greedy
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X O O |
2 |O O X O O O |
3 |O O O O X O |
4 |O O O O O O |
5 |- O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X X O |
2 |X X X X X O |
3 |X X X X X O |
4 |X X X X X O |
5 |X X X X - O |
-----------------------
(2, 0, 0)




[PITTING] alphabeta-simple vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |X X X O O O |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X O O X |
4 |X X O O X X |
5 |X O O O O X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X O O O |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X O O X |
4 |X X O O X X |
5 |X O O O O X |
-----------------------
(1, 1, 0)




[PITTING] alphabeta-simple vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O X O X O X |
2 |X X X O X X |
3 |X X O O O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O X O O |
2 |O O O X X O |
3 |O O O O X O |
4 |O O O O X O |
5 |O O O O X O |
-----------------------
(0, 2, 0)




[PITTING] alphabeta-simple vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O X X X X O |
2 |O X O X X O |
3 |O X O O X O |
4 |O X X X X O |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] alphabeta-simple vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] alphabeta-strategy vs random
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X O O O O |
2 |X X X X O O |
3 |O O O O O O |
4 |O O O O O O |
5 |O X O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X O X X X |
3 |X X X X X X |
4 |O X X X X X |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] alphabeta-strategy vs greedy
   0 1 2 3 4 5 
-----------------------
0 |O O O O X X |
1 |O O O O O - |
2 |O O O O O O |
3 |O O X O O O |
4 |O O O O X O |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X - X |
1 |X X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X - X X - X |
-----------------------
(2, 0, 0)




[PITTING] alphabeta-strategy vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O X O O |
2 |O O O X X O |
3 |O O O O X O |
4 |O O O O X O |
5 |O O O O X O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O X O X O X |
2 |X X X O X X |
3 |X X O O O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] alphabeta-strategy vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O X O X X |
2 |O X X X O X |
3 |O O X O X X |
4 |O X X X X X |
5 |O X X O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O X O X X |
2 |O X X X O X |
3 |O O X O X X |
4 |O X X X X X |
5 |O X X O O O |
-----------------------
(1, 1, 0)




[PITTING] alphabeta-strategy vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X O X |
2 |O X X O O X |
3 |O O X O X X |
4 |O O O X X X |
5 |O O O O X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O O X |
2 |O X X O O X |
3 |O X O X O X |
4 |O X X X O X |
5 |O X X X X X |
-----------------------
(1, 1, 0)




[PITTING] alphabeta-strategy vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |X O X O O O |
2 |X X O X O O |
3 |X O X O O O |
4 |X X O X O O |
5 |X X X X X X |
-----------------------
(0, 2, 0)




[PITTING] nnet-200 vs random
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O X X X X O |
2 |O O X O O O |
3 |O X O X X O |
4 |O O O X X O |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X O O O O X |
1 |X X X O X X |
2 |X X O X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X O X X X X |
-----------------------
(2, 0, 0)




[PITTING] nnet-200 vs greedy
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] nnet-200 vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] nnet-200 vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O O |
2 |O O X X O X |
3 |O X O O X X |
4 |X X X X X X |
5 |O O O O X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
(1, 1, 0)




[PITTING] nnet-200 vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O X X X |
2 |X O X X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O X X X |
2 |X O X X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
(1, 1, 0)




[PITTING] nnet-200 vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X O O O O X |
2 |X X X O X X |
3 |X X O X X X |
4 |X O O O X X |
5 |O X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O X |
2 |O X O O X X |
3 |O X O X O O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
(0, 2, 0)




[PITTING] nnet-500 vs random
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O O |
2 |O O O O O X |
3 |O O O O O O |
4 |O O O O O O |
5 |X X X O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X O X X |
3 |X X X O X X |
4 |X O X X X X |
5 |X X X X X O |
-----------------------
(2, 0, 0)




[PITTING] nnet-500 vs greedy
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O X O |
2 |O O O X X O |
3 |X X X X X O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X O O X X X |
2 |X X O X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X X X X X O |
-----------------------
(2, 0, 0)




[PITTING] nnet-500 vs alphabeta-simple
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
(2, 0, 0)




[PITTING] nnet-500 vs alphabeta-strategy
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O - |
2 |O O O O X X |
3 |O O X X X X |
4 |O X X O X X |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
(2, 0, 0)




[PITTING] nnet-500 vs nnet-200
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O X |
2 |O X O O X X |
3 |O X O X O O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X O O X X X |
2 |X O X X X X |
3 |X X X X O X |
4 |X X X O O X |
5 |O O O O O X |
-----------------------
(2, 0, 0)




[PITTING] nnet-500 vs nnet-500
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X O O O O |
2 |X X X O X O |
3 |X X X X X O |
4 |X X O X X O |
5 |X X X X X X |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X O O X |
2 |O X O O O X |
3 |O O O X O X |
4 |O O O O X X |
5 |O O O O O X |
-----------------------
(0, 2, 0)
```

