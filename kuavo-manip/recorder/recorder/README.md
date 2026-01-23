# Recorder
This tool records ROS bags and detects missing or low-frequency topics after recording.

## Usage

You can use following cmd line or examples provided to record the data you need and they will be stored as a structure like `/raw_data/episode_x.bag`
```bash
# Start interactive recording control
python record.py -n <task_name>

# The script will show:
# - Press 'c' to start recording
# - Press 's' to stop recording
# - Press 'q' to quit
```