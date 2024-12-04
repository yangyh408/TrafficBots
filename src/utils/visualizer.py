import os
import io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from waymo_open_dataset.utils.sim_agents import visualizations
from utils.transform_utils import torch_pos2global, torch_rad2rot, torch_rad2global
from utils.opendrive2discretenet.parser import parse_opendrive

def plot_xodr_map(ax, xodr_path, draw_arrow: bool=False) -> None:

    road_info = parse_opendrive(xodr_path)

    """根据parse_opendrive模块解析出的opendrive路网信息绘制道路"""
    xlim1 = float("Inf")
    xlim2 = -float("Inf")
    ylim1 = float("Inf")
    ylim2 = -float("Inf")
    color = "gray"
    label = None

    for discrete_lane in road_info.discretelanes:
        verts = []
        codes = [Path.MOVETO]

        for x, y in np.vstack(
            [discrete_lane.left_vertices, discrete_lane.right_vertices[::-1]]
        ):
            verts.append([x, y])
            codes.append(Path.LINETO)

            # if color != 'gray':
            xlim1 = min(xlim1, x)
            xlim2 = max(xlim2, x)

            ylim1 = min(ylim1, y)
            ylim2 = max(ylim2, y)

        verts.append(verts[0])
        codes[-1] = Path.CLOSEPOLY

        path = Path(verts, codes)

        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor=color,
                edgecolor="black",
                lw=0.0,
                alpha=0.5,
                zorder=0,
                label=label,
            )
        )

        ax.plot(
            [x for x, _ in discrete_lane.left_vertices],
            [y for _, y in discrete_lane.left_vertices],
            color="black",
            lw=0.3,
            zorder=1,
        )
        ax.plot(
            [x for x, _ in discrete_lane.right_vertices],
            [y for _, y in discrete_lane.right_vertices],
            color="black",
            lw=0.3,
            zorder=1,
        )

        ax.plot(
            [x for x, _ in discrete_lane.center_vertices],
            [y for _, y in discrete_lane.center_vertices],
            color="white",
            alpha=0.5,
            lw=0.8,
            zorder=1,
        )

        if draw_arrow:
            mc = discrete_lane.center_vertices
            total_len = ((mc[0][0] - mc[-1][0]) ** 2 + (mc[0][1] - mc[-1][1]) ** 2) ** 0.5
            if total_len > 30:
                index_ = list(map(int, np.linspace(start=10, stop=mc.shape[0] - 10, num=4)))
            else:
                index_ = []
            for i in range(len(index_)):
                start_c, end_c = mc[index_[i]], mc[index_[i] + 1]
                ax.arrow(
                    start_c[0], start_c[1], end_c[0] - start_c[0], end_c[1] - start_c[1],
                    shape='full',
                    color='white',
                    alpha=0.5,
                    head_width=1,
                    head_length=2,
                    length_includes_head=True,
                    zorder=1,
                )
    ax.set_xlim(xlim1, xlim2)
    ax.set_ylim(ylim1, ylim2)
    return

class Visualizer:
    def __init__(self, scenario, scenario_center, scenario_yaw, activate: bool=False):
        self.activate = activate

        if self.activate:
            self.frames = []
            self.scenario_center = scenario_center
            self.scenario_yaw = scenario_yaw
            # 创建画布
            self.fig, self.ax_map_bg = plt.subplots(1, 1, figsize=(10, 10))
            # 创建动态元素图层
            self.ax_objects = self.ax_map_bg.twinx()

            # 绘制静态地图
            self._plot_static_map(self.ax_map_bg, scenario)

            plt.ion()
        
    @staticmethod
    def _plot_static_map(ax, scenario):
        visualizations.add_map(ax, scenario)

    def update(self, preds, valid):
        if self.activate:
            self.ax_objects.cla()
            self.ax_objects.set_xlim(self.ax_map_bg.get_xlim())
            self.ax_objects.set_ylim(self.ax_map_bg.get_ylim())
            self.ax_objects.axis('off')

            sim_pos_global = torch_pos2global(preds[..., :2], self.scenario_center.unsqueeze(1), torch_rad2rot(self.scenario_yaw))[0]
            sim_yaw_global = torch_rad2global(preds[..., 2:3], self.scenario_yaw)[0]

            for agent_id in range(sim_pos_global.shape[0]):
                if valid[0, agent_id]:
                    if agent_id == 0:
                        self.ax_objects.plot(sim_pos_global[agent_id, 0], sim_pos_global[agent_id, 1], 'o', markersize=5, c='#bf77f6')
                    else:
                        self.ax_objects.plot(sim_pos_global[agent_id, 0], sim_pos_global[agent_id, 1], 'o', markersize=5, c='#75bbfd')
                    self.ax_objects.plot([sim_pos_global[agent_id, 0], sim_pos_global[agent_id, 0] + np.cos(sim_yaw_global[agent_id]) * 2], 
                                         [sim_pos_global[agent_id, 1], sim_pos_global[agent_id, 1] + np.sin(sim_yaw_global[agent_id]) * 2], '-', linewidth=1, color='#653700')
                    self.ax_objects.text(sim_pos_global[agent_id, 0], sim_pos_global[agent_id, 1], str(agent_id), ha='center', va='center', fontsize=6, color='black')
            
            # 保存当前帧为图片  
            buf = io.BytesIO()  
            plt.savefig(buf, format='png')  
            buf.seek(0)  
            self.frames.append(imageio.imread(buf))  
            
            plt.pause(1e-7)

    def close(self, save_path: str=""):
        if self.activate:
            plt.ioff()
            plt.close()

            if save_path:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                imageio.mimsave(save_path, self.frames, fps=10)

