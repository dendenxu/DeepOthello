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
(base)  ✘  ~/OneDrive - zju.edu.cn/alpha-zero-general-master   master ●  python pit.py
random vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O X |
1 |X O O O X X |
2 |X X O O X X |
3 |X X O O X X |
4 |X O X X O X |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 30.20it/s]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O O X X |
2 |X X X O O X |
3 |X X X X O X |
4 |X X O O X X |
5 |X X O O X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 33.34it/s]
(2, 0, 0)
random vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X O O O O |
1 |X X X O O X |
2 |X X O X O X |
3 |X X O O X X |
4 |X X O X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 24.88it/s]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |O X O O X O |
2 |O X O X X O |
3 |O X X X X O |
4 |O X X O O O |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 18.25it/s]
(0, 2, 0)
random vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X - |
2 |X X X X X O |
3 |X X X X X - |
4 |X X X X X X |
5 |X X X X X - |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:08<00:00,  8.72s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  37 Result  1
   0 1 2 3 4 5 
-----------------------
0 |X X O O O O |
1 |O X O O O O |
2 |O O X O O O |
3 |X X O O O O |
4 |O O O O O O |
5 |O O O O - O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:06<00:00,  6.49s/it]
(0, 2, 0)
random vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O - X X X X |
1 |O X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X X X X O X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:18<00:00, 18.02s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X X O O |
2 |O X O O X O |
3 |O O X X O O |
4 |O X O X X O |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:15<00:00, 15.09s/it]
(0, 2, 0)
random vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |O X X X O X |
2 |O O X X O X |
3 |X X O X O X |
4 |X X O O X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:57<00:00, 57.42s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X X O |
2 |O X O X O O |
3 |O O O O O O |
4 |O O O O O X |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:02<00:00, 62.68s/it]
(0, 2, 0)
random vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X O X X X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:45<00:00, 105.15s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O O |
2 |O O X X O X |
3 |O O X O X X |
4 |X O O X X X |
5 |X O O O O O |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [02:17<00:00, 137.80s/it]
(0, 2, 0)
greedy vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X O X |
1 |X X X O O O |
2 |X X O X O O |
3 |X X O X O O |
4 |X X X O O O |
5 |X X O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 29.48it/s]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X O O |
2 |X X X O X O |
3 |X X O X X O |
4 |X X X O X O |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 29.16it/s]
(1, 1, 0)
greedy vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  35 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X X X O O |
3 |X X X O O O |
4 |X X O X O O |
5 |X O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 24.59it/s]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  35 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X X X O O |
3 |X X X O O O |
4 |X X O X O O |
5 |X O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:00<00:00, 23.31it/s]
(1, 1, 0)
greedy vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X X O |
2 |X X X X X O |
3 |X X X X X O |
4 |X X X X X O |
5 |X X X X - O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:07<00:00,  7.84s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X O O |
2 |O O X O O O |
3 |O O O O X O |
4 |O O O O O O |
5 |- O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:09<00:00,  9.33s/it]
(0, 2, 0)
greedy vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  36 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X - X |
1 |X X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X - X X - X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:08<00:00,  8.73s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  31 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O X X |
1 |O O O O O - |
2 |O O O O O O |
3 |O O X O O O |
4 |O O O O X O |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:11<00:00, 11.15s/it]
(0, 2, 0)
greedy vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O O X X X |
3 |X O O X X X |
4 |X O O O X X |
5 |X X X X O X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:59<00:00, 59.01s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  37 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:56<00:00, 56.34s/it]
(0, 2, 0)
greedy vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  36 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:57<00:00, 117.45s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  37 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [02:01<00:00, 121.02s/it]
(0, 2, 0)
alphabeta-simple vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  27 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O - |
2 |O O O O - O |
3 |- O O O O O |
4 |O O O O O O |
5 |- - - O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:08<00:00,  8.33s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O X X X X |
1 |X X O X O X |
2 |X X X O O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:06<00:00,  6.73s/it]
(2, 0, 0)
alphabeta-simple vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  33 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X X O O |
2 |O O X O O O |
3 |O O O O X O |
4 |O O O O O O |
5 |- O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:09<00:00,  9.35s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X O |
1 |X X X X X O |
2 |X X X X X O |
3 |X X X X X O |
4 |X X X X X O |
5 |X X X X - O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:07<00:00,  7.66s/it]
(2, 0, 0)
alphabeta-simple vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X O O O |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X O O X |
4 |X X O O X X |
5 |X O O O O X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:14<00:00, 14.07s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X O O O |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X O O X |
4 |X X O O X X |
5 |X O O O O X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:14<00:00, 14.11s/it]
(1, 1, 0)
alphabeta-simple vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O X O X O X |
2 |X X X O X X |
3 |X X O O O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:23<00:00, 23.72s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O X O O |
2 |O O O X X O |
3 |O O O O X O |
4 |O O O O X O |
5 |O O O O X O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:22<00:00, 22.18s/it]
(0, 2, 0)
alphabeta-simple vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:03<00:00, 63.13s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O X X X X O |
2 |O X O X X O |
3 |O X O O X O |
4 |O X X X X O |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:14<00:00, 74.65s/it]
(0, 2, 0)
alphabeta-simple vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:56<00:00, 116.88s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [02:22<00:00, 142.55s/it]
(0, 2, 0)
alphabeta-strategy vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X O O O O |
2 |X X X X O O |
3 |O O O O O O |
4 |O O O O O O |
5 |O X O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:13<00:00, 13.02s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  36 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X O |
2 |X X O X X X |
3 |X X X X X X |
4 |O X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:27<00:00, 27.15s/it]
(2, 0, 0)
alphabeta-strategy vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  31 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O X X |
1 |O O O O O - |
2 |O O O O O O |
3 |O O X O O O |
4 |O O O O X O |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:10<00:00, 10.95s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  36 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X - X |
1 |X X X X X X |
2 |X X X X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X - X X - X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:08<00:00,  8.64s/it]
(2, 0, 0)
alphabeta-strategy vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O X O O |
2 |O O O X X O |
3 |O O O O X O |
4 |O O O O X O |
5 |O O O O X O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:22<00:00, 22.31s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O X O X O X |
2 |X X X O X X |
3 |X X O O O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:23<00:00, 23.53s/it]
(2, 0, 0)
alphabeta-strategy vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O X O X X |
2 |O X X X O X |
3 |O O X O X X |
4 |O X X X X X |
5 |O X X O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:22<00:00, 22.56s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O X O X X |
2 |O X X X O X |
3 |O O X O X X |
4 |O X X X X X |
5 |O X X O O O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:22<00:00, 22.55s/it]
(1, 1, 0)
alphabeta-strategy vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X O X |
2 |O X X O O X |
3 |O O X O X X |
4 |O O O X X X |
5 |O O O O X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:01<00:00, 61.23s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O O X |
2 |O X X O O X |
3 |O X O X O X |
4 |O X X X O X |
5 |O X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:39<00:00, 99.40s/it]
(1, 1, 0)
alphabeta-strategy vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:49<00:00, 109.74s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |X O X O O O |
2 |X X O X O O |
3 |X O X O O O |
4 |X X O X O O |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [03:01<00:00, 181.28s/it]
(0, 2, 0)
nnet-200 vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O X X X X O |
2 |O O X O O O |
3 |O X O X X O |
4 |O O O X X O |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:56<00:00, 56.20s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O X |
1 |X X X O X X |
2 |X X O X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X O X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:46<00:00, 46.91s/it]
(2, 0, 0)
nnet-200 vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  37 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O X |
1 |O O O O X X |
2 |O O O X O X |
3 |O O O X O X |
4 |O O O O O X |
5 |O O O O O X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [00:45<00:00, 45.92s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  36 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X X O X |
3 |X X X X O X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:45<00:00, 45.90s/it]
(2, 0, 0)
nnet-200 vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:02<00:00, 62.52s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:53<00:00, 53.01s/it]
(2, 0, 0)
nnet-200 vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O O |
2 |O O X X O X |
3 |O X O O X X |
4 |X X X X X X |
5 |O O O O X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:28<00:00, 88.67s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:43<00:00, 43.12s/it]
(1, 1, 0)
nnet-200 vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O X X X |
2 |X O X X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:04<00:00, 64.54s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X O X X X |
2 |X O X X X X |
3 |X O X X X X |
4 |X X X X X X |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [00:59<00:00, 59.46s/it]
(1, 1, 0)
nnet-200 vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X O O O O X |
2 |X X X O X X |
3 |X X O X X X |
4 |X O O O X X |
5 |O X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [02:53<00:00, 173.58s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O X |
2 |O X O O X X |
3 |O X O X O O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [02:47<00:00, 167.12s/it]
(0, 2, 0)
nnet-500 vs random
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  35 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O O |
2 |O O O O O X |
3 |O O O O O O |
4 |O O O O O O |
5 |X X X O O O |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:51<00:00, 111.15s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |X X X O X X |
3 |X X X O X X |
4 |X O X X X X |
5 |X X X X X O |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [01:56<00:00, 116.78s/it]
(2, 0, 0)
nnet-500 vs greedy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O X O |
2 |O O O X X O |
3 |X X X X X O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:54<00:00, 114.07s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  34 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X O O X X X |
2 |X X O X X X |
3 |X X X X X X |
4 |X X X X X X |
5 |X X X X X O |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:33<00:00, 93.77s/it]
(2, 0, 0)
nnet-500 vs alphabeta-simple
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |X X X O X O |
2 |X X X X O O |
3 |X X O O X O |
4 |X X O X X X |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|████████████████████████████████████████| 1/1 [01:10<00:00, 70.79s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X X X X X X |
1 |X X X X X X |
2 |O O X O X X |
3 |X O O X X X |
4 |X X X X X O |
5 |X X X X X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:28<00:00, 88.70s/it]
(2, 0, 0)
nnet-500 vs alphabeta-strategy
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  31 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O O O O - |
2 |O O O O X X |
3 |O O X X X X |
4 |O X X O X X |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [02:21<00:00, 141.05s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X X X X X |
2 |O X X X X X |
3 |O O X O X X |
4 |O X O X X X |
5 |O O O O X X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:20<00:00, 80.29s/it]
(2, 0, 0)
nnet-500 vs nnet-200
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O X X X X X |
1 |O O O O O X |
2 |O X O O X X |
3 |O X O X O O |
4 |O O X X O O |
5 |O O O O O O |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [01:59<00:00, 119.26s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X O O X X X |
2 |X O X X X X |
3 |X X X X O X |
4 |X X X O O X |
5 |O O O O O X |
-----------------------
Arena.playGames (2): 100%|███████████████████████████████████████| 1/1 [01:46<00:00, 106.29s/it]
(2, 0, 0)
nnet-500 vs nnet-500
Arena.playGames (1):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  32 Result  -1
   0 1 2 3 4 5 
-----------------------
0 |X O O O O O |
1 |X X O O O O |
2 |X X X O X O |
3 |X X X X X O |
4 |X X O X X O |
5 |X X X X X X |
-----------------------
Arena.playGames (1): 100%|███████████████████████████████████████| 1/1 [02:02<00:00, 122.36s/it]
Arena.playGames (2):   0%|                                                | 0/1 [00:00<?, ?it/s]Game over: Turn  35 Result  1
   0 1 2 3 4 5 
-----------------------
0 |O O O O O O |
1 |O O X O O X |
2 |O X O O O X |
3 |O O O X O X |
4 |O O O O X X |
5 |O O O O O X |
-----------------------
Arena.playGames (2): 100%|████████████████████████████████████████| 1/1 [01:30<00:00, 90.05s/it]
(0, 2, 0)
(base)  ~/OneDrive - zju.edu.cn/alpha-zero-general-master   master ●  
```

