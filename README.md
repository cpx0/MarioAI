# Mario AI: Train a Mario-playing RL Agent

## Build the RL Environment with Docker

### Build: Docker Image 'dm_control:1.1'

```bash
cd docker
. build.sh
```

### Run: Docker Container 'mario_ai'

```bash
cd ..
. launch.sh mario_ai
```

## Train a Mario-playing RL Agent

1. Open http://127.0.0.1:9280 at your browser.
2. When 'Build Recommended' window pops up, click 'Cancel' button.
3. Open 'Terminal' in 'Launcher'.
4. Follow commands in the terminal to train a mario-playing rl agent.

```bush
cd ~/workspace
python train.py
```


