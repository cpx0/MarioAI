# Mario AI: Train a Mario-playing RL Agent

## Build the RL Environment with Docker
![docker image_ci workflow](https://github.com/cpx0/MarioAI/actions/workflows/docker-image.yml/badge.svg)

### Build: Docker Image 'dm_control:1.2'

```bash
cd docker
./build.sh
```

### Launch: Docker Container 'mario_ai'

```bash
cd ..
./launch.sh
```

## Train a Mario-playing RL Agent

1. Open http://127.0.0.1:9280 at your browser.
2. When 'Build Recommended' window pops up, click 'Cancel' button.
3. Open 'Terminal' in 'Launcher'.
4. Follow commands in the terminal to train a mario-playing rl agent.

```bush
cd ~/workspace
sudo python train.py
```

## Play Mario with a Trained RL Agent

```bush
cd ~/workspace
sudo python demo.py
```
