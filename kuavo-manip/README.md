# Kuavo-Manip(beta) : IL With Leobot Pipeline

This repo provides a pipeline of data collecting.

## Complete Workflow of Kuavo-Manip

### Datasets

You can follow the examples of collector to make your datasets by robot `kuavo`.


#### 1. install local dependencies
``` bash
pip3 install -r requirements.txt
```

#### 2. Run data collection scripts

- Use `-i` to request user input for each episode (instead of only once at startup).

- Select the robot platform with `--platform` (default: `wheeled`).

- Set the number of episodes with `--episodes` (default: `50`).

```bash
python3 collector/runner.py -i -n your_task_name --platform legged --episodes 100
```
