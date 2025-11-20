import time

import rerun as rr
import numpy as np
import torch

def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    if isinstance(chw_float32_torch, np.ndarray):
        print(np.shape(chw_float32_torch))
        chw_float32_torch = torch.from_numpy(chw_float32_torch)

    print(chw_float32_torch.ndim)
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy

class RerunVisualizer:
    def __init__(self):
        rr.init("viz2", spawn=True)
        self.control_frequency = 10.0  # 控制频率(Hz)
        self.control_dt = 1.0 / self.control_frequency

    def visualize_chunk(self, name, chunk_data, step_id=0, width=None, x_axis=None, color=None):
        """
        step_id: chunk起始时候的step索引
        """
        x_name = "step"
        if width is not None or color is not None:
            series_kwargs = {}
            if width is not None:
                series_kwargs["widths"] = width
            if color is not None:
                series_kwargs["colors"] = np.array([color])
            rr.log(name, rr.SeriesLines(**series_kwargs), static=True)
        # if isiterable(chunk_data):
        # 判断是否是list
        if isinstance(chunk_data, (list, tuple, np.ndarray)):
            for local_step, data in enumerate(chunk_data):
                if x_axis is not None:
                    global_step = x_axis[local_step + step_id]
                else:
                    global_step = local_step + step_id
                rr.set_time(timeline=x_name, sequence=int(global_step))
                y = data
                rr.log(name, rr.Scalars(y))
        else:
            if x_axis is not None:
                global_step = x_axis[0 + step_id]
            else:
                global_step = 0 + step_id
            rr.set_time(timeline=x_name, sequence=int(global_step))
            y = chunk_data
            rr.log(name, rr.Scalars(y))

    def del_chunk(self, name, chunk_data, step_id=0, width=None):
        x_name = "step"
        if width is not None:
            rr.log(name, rr.SeriesLines(widths=width), static=True)
        for local_step, data in enumerate(chunk_data):
            global_step = local_step + step_id
            rr.set_time(timeline=x_name, sequence=int(global_step))
            rr.log(name, rr.Clear(recursive=True))

    def clear_path(self, path: str):
        rr.log(path, rr.Clear(recursive=True))

    def show_img(self, name, image_data, step_id=0):
        # batch[key][i]
        x_name = "step"
        rr.set_time(timeline=x_name, sequence=int(step_id))
        rr.log(name, rr.Image(to_hwc_uint8_numpy(image_data)))

    def visualize_points(self, name, xs, ys, step_axis=None, colors=None):
        x_name = "step"
        xs = np.asarray(xs).reshape(-1)
        ys = np.asarray(ys).reshape(-1)
        if xs.shape != ys.shape:
            raise ValueError("xs and ys must have the same shape")
        if step_axis is not None:
            xs = step_axis[xs]
        for x_val, y_val in zip(xs, ys):
            rr.set_time(timeline=x_name, sequence=int(x_val))
            rr.log(name, rr.Scalars(float(y_val)))


from pynput import keyboard

class KeyboardManager:
    def __init__(self):
        self.paused = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.space:    # 空格：切换暂停
            self.paused = not self.paused
            print("paused" if self.paused else "resumed")
        elif key == keyboard.Key.esc:    # ESC：退出监听线程
            return False

def try_visualize_chunk():
    visualizer = RerunVisualizer()
    # actions dim (120, 14)
    # 创建sin cos的假的action
    eps_len = 120
    chunk_size = 20
    action_dim = 14
    # 真实执行的actions
    gt_actions = np.zeros([eps_len, action_dim])

    km = KeyboardManager()

    for dim in range(action_dim):
        for i in range(eps_len):
            gt_actions[i, dim] = np.sin(i / eps_len * 2 * np.pi * (dim + 1))

    for dim in range(action_dim):
        visualizer.visualize_chunk(
            name=f"chunk/action_dim_{dim}/gt",
            chunk_data=gt_actions[:, dim],
            step_id=0,
            width=3.0
        )

    # 模型预测的action_chunk
    last_eps_step = 0
    for eps_step in range(eps_len):
        while km.paused:
            time.sleep(0.1)
        # 每次重新伪造一个chunk
        time.sleep(0.1)
        pred_chunk = np.zeros([chunk_size, action_dim])
        for dim in range(action_dim):
            for i in range(chunk_size):
                pred_chunk[i, dim] = np.sin((eps_step + i) / eps_len * 2 * np.pi * (dim + 1) + 0.1)

        # 可视化此chunk并且清除上一个chunk
        for dim in range(action_dim):
            visualizer.visualize_chunk(
                name=f"chunk/action_dim_{dim}/pred_seg_{eps_step}",
                chunk_data=pred_chunk[:, dim],
                step_id=eps_step,
                width=2
            )
            visualizer.del_chunk(
                name=f"chunk/action_dim_{dim}/pred_seg_{last_eps_step}",
                chunk_data=pred_chunk[:, dim],
                step_id=last_eps_step,
                width=0.5
            )

        last_eps_step = eps_step



if __name__ == "__main__":
    # rr.init("viz0", spawn=True)
    try_visualize_chunk()
    # rr.flush()
