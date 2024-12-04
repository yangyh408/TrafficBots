import os 
import math
import torch
import tensorflow as tf
import numpy as np 
from lxml import etree
from xml.dom.minidom import parse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from controller.manual import ManualController
from controller.idm import IDM
import pl_modules.waymo_motion as waymo_motion_module
import utils.pack_h5 as pack_utils
from utils.opendrive2discretenet.network import Network
from utils.opendrive2discretenet.opendriveparser.parser import parse_opendrive as parse_opendrive_xml
from utils.transform_utils import torch_pos2global, torch_rad2rot
from utils.visualizer import plot_xodr_map

class AgentInfo:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type
        self.length = 0.0
        self.width = 0.0
        self.height = 0.0
        # t: [center_x, center_y, center_z, heading, velocity_x, velocity_y]
        self.traj = {}
    
    def set_ego_traj(self, x, y, h, v):
        v = abs(v)
        h = self._normalize_h(h)
        self.traj[0.0] = np.array([x, y, 0.0, h, v * math.cos(h), v * math.sin(h)])
    
    def set_other_traj(self, times, trajs):
        for time, traj in zip(times, trajs):
            self.traj[time] = traj
    
    # [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    def get_traj_info(self, start_time, end_time, dt = 0.1):
        trajs = []
        t = start_time
        while (t <= end_time):
            cur_state = self.traj.get(t, None)
            if cur_state is not None:
                trajs.append([
                    cur_state[0], cur_state[1], cur_state[2],
                    self.length, self.width, self.height,
                    cur_state[3], cur_state[4], cur_state[5], 1
                ])
            else:
                trajs.append([
                    0.0, 0.0, 0.0, self.length, self.width, self.height, 0.0, 0.0, 0.0, 0
                ])
            t += dt
        return trajs

    @staticmethod
    def _normalize_h(h):
        return ((h % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi)

def extract_map_from_xodr(xodr_path):
    mf_id = []
    mf_xyz = []
    mf_type = []
    mf_edge = []
    id_map = {}

    with open(xodr_path, 'r', encoding='utf-8') as fh:
        root = etree.parse(fh).getroot()
    openDriveXml = parse_opendrive_xml(root)

    loadedRoadNetwork = Network()
    loadedRoadNetwork.load_opendrive(openDriveXml)

    for id, parametric_lane in enumerate(loadedRoadNetwork._planes):
        if parametric_lane.type == "driving":
            id_map[parametric_lane.id_] = id
            mf_id.append(id)

            discrete_lane = parametric_lane.to_discretelane()
            mf_xyz.append(
                np.concatenate((discrete_lane.center_vertices, 
                                np.zeros((discrete_lane.center_vertices.shape[0], 1))), axis=1)
            )
            mf_type.append(1)

    for id, parametric_lane in enumerate(loadedRoadNetwork._planes):
        if parametric_lane.type == "driving":
            successor = loadedRoadNetwork._link_index.get_successors(parametric_lane.id_)
            for old_id in successor:
                mf_edge.append([id, id_map.get(old_id, -1)])
    return mf_id, mf_xyz, mf_type, mf_edge

def extract_light_from_json():
    tl_lane_state = [[] for _ in range(11)]
    tl_lane_id = [[] for _ in range(11)]
    tl_stop_point = [[] for _ in range(11)]
    return tl_lane_state, tl_lane_id, tl_stop_point

def extract_agent_from_xosc(xosc_path):
    agents = {}

    opens = parse(xosc_path).documentElement
    objects = opens.getElementsByTagName('ScenarioObject')
    for idx, object_ in enumerate(objects):
        agent_name = object_.getAttribute("name")
        category_elements = object_.getElementsByTagName('Vehicle')
        if category_elements:
            agent_type = category_elements[0].getAttribute("vehicleCategory")
        else:
            agent_type = "pedestrian"
            
        agent = AgentInfo(idx, agent_name, agent_type)
        dimension_element = object_.getElementsByTagName('Dimensions')[0]
        for k, v in dimension_element.attributes.items():
            agent.__setattr__(k, float(v))

        agents[agent_name] = agent
        
    # 读取本车信息
    ego_node = opens.getElementsByTagName('Private')[0]
    ego_init = ego_node.childNodes[3].data
    ego_v, ego_x, ego_y, ego_head = [float(i.split('=')[1]) for i in ego_init.split(',')]
    agents['Ego'].set_ego_traj(ego_x, ego_y, ego_head, ego_v)

    # 读取背景车信息
    select_keys = ['x', 'y', 'z', 'h']
    acts = opens.getElementsByTagName('Act')
    for act in acts:
        agent_name = act.getAttribute("name").split('_')[-1]

        times = []
        trajs = []
        for status in act.getElementsByTagName('Vertex'):
            loc = status.getElementsByTagName('WorldPosition')[0]
            times.append(round(float(status.getAttribute('time')), 3))
            trajs.append([loc.getAttribute(key) for key in ['x', 'y', 'z', 'h']])
        
        times = np.array(times, dtype=float)
        trajs = np.array(trajs, dtype=float)
        trajs = np.concatenate([
                trajs[:-1], 
                (np.diff(trajs[:, 0]) / np.diff(times))[:, None], 
                (np.diff(trajs[:, 1]) / np.diff(times))[:, None]
            ], 
            axis=-1
        )
        agents[agent_name].set_other_traj(times, trajs)

    TYPES = ['car', 'pedestrian', 'bicycle']
    agent_id = []
    agent_type = []
    agent_role = []
    agent_states = []
    for agent in agents.values():
        agent_id.append(agent.id)
        agent_type.append(TYPES.index(agent.type))
        if agent.name == 'Ego':
            agent_role.append([True, False, False])
        else:
            agent_role.append([False, False, False])
        agent_states.append(agent.get_traj_info(0, 1))

    return agent_id, agent_type, agent_states, agent_role

def generate_batch_info(xodr_path, xosc_path, scenario_id):
    mf_id, mf_xyz, mf_type, mf_edge = extract_map_from_xodr(xodr_path)
    tl_lane_state, tl_lane_id, tl_stop_point = extract_light_from_json()
    agent_id, agent_type, agent_states, agent_role = extract_agent_from_xosc(xosc_path)

    N_PL_TYPE = 11
    DIM_VEH_LANES = [0, 1, 2]
    DIM_CYC_LANES = [3]
    DIM_PED_LANES = [4]

    N_TL_STATE = 5

    N_PL_MAX = 3000
    N_TL_MAX = 40
    N_AGENT_MAX = 1300

    N_PL = 1024
    N_TL = 100
    N_AGENT = 64
    N_AGENT_NO_SIM = 256

    THRESH_MAP = 500  # ! 120
    THRESH_AGENT = 120

    N_STEP = 91
    STEP_CURRENT = 10

    pack_all = False
    pack_history = True

    episode = {}
    n_pl = pack_utils.pack_episode_map(
        episode=episode, mf_id=mf_id, mf_xyz=mf_xyz, mf_type=mf_type, mf_edge=mf_edge, n_pl_max=N_PL_MAX
    )
    n_tl = pack_utils.pack_episode_traffic_lights(
        episode=episode,
        tl_lane_state=tl_lane_state,
        tl_lane_id=tl_lane_id,
        tl_stop_point=tl_stop_point,
        pack_all=pack_all,
        pack_history=pack_history,
        n_tl_max=N_TL_MAX,
        step_current=STEP_CURRENT,
    )
    n_agent = pack_utils.pack_episode_agents(
        episode=episode,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_states=agent_states,
        agent_role=agent_role,
        pack_all=pack_all,
        pack_history=pack_history,
        n_agent_max=N_AGENT_MAX,
        step_current=STEP_CURRENT,
    )
    scenario_center, scenario_yaw = pack_utils.center_at_sdc(episode)

    episode_reduced = {}
    pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=5)
    episode_with_map = episode["map/valid"].any(1).sum() > 0
    pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

    pack_utils.filter_episode_traffic_lights(episode)
    pack_utils.repack_episode_traffic_lights(episode, episode_reduced, N_TL, N_TL_STATE)

    mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
        episode=episode,
        episode_reduced=episode_reduced,
        n_agent=N_AGENT,
        prefix="history/",
        dim_veh_lanes=DIM_VEH_LANES,
        dist_thresh_agent=THRESH_AGENT,
        step_current=STEP_CURRENT,
    )
    episode_reduced["map/boundary"] = pack_utils.get_map_boundary(
        episode_reduced["map/valid"], episode_reduced["map/pos"]
    )

    pack_utils.repack_episode_agents(episode, episode_reduced, mask_sim, N_AGENT, "history/")
    pack_utils.repack_episode_agents_no_sim(
        episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
    )

    n_step = 91
    n_step_history = 11
    n_agent = 64
    n_agent_no_sim = 256
    n_pl = 1024
    n_tl = 100
    n_tl_stop = 40
    n_pl_node = 20

    tensor_size_test = {
        # object_id for waymo metrics
        "history/agent/object_id": (n_agent,),
        "history/agent_no_sim/object_id": (n_agent_no_sim,),
        # agent_sim
        "history/agent/valid": (n_step_history, n_agent),  # bool,
        "history/agent/pos": (n_step_history, n_agent, 2),  # float32
        "history/agent/z": (n_step_history, n_agent, 1),  # float32
        "history/agent/vel": (n_step_history, n_agent, 2),  # float32, v_x, v_y
        "history/agent/spd": (n_step_history, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
        "history/agent/acc": (n_step_history, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
        "history/agent/yaw_bbox": (n_step_history, n_agent, 1),  # float32, yaw of the bbox heading
        "history/agent/yaw_rate": (n_step_history, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
        "history/agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
        "history/agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
        "history/agent/size": (n_agent, 3),  # float32: [length, width, height]
        "history/agent_no_sim/valid": (n_step_history, n_agent_no_sim),
        "history/agent_no_sim/pos": (n_step_history, n_agent_no_sim, 2),
        "history/agent_no_sim/z": (n_step_history, n_agent_no_sim, 1),
        "history/agent_no_sim/vel": (n_step_history, n_agent_no_sim, 2),
        "history/agent_no_sim/spd": (n_step_history, n_agent_no_sim, 1),
        "history/agent_no_sim/yaw_bbox": (n_step_history, n_agent_no_sim, 1),
        "history/agent_no_sim/type": (n_agent_no_sim, 3),
        "history/agent_no_sim/size": (n_agent_no_sim, 3),
        # map
        "map/valid": (n_pl, n_pl_node),  # bool
        "map/type": (n_pl, 11),  # bool one_hot
        "map/pos": (n_pl, n_pl_node, 2),  # float32
        "map/dir": (n_pl, n_pl_node, 2),  # float32
        "map/boundary": (4,),  # xmin, xmax, ymin, ymax
        # traffic_light
        "history/tl_lane/valid": (n_step_history, n_tl),  # bool
        "history/tl_lane/state": (n_step_history, n_tl, 5),  # bool one_hot
        "history/tl_lane/idx": (n_step_history, n_tl),  # int, -1 means not valid
        "history/tl_stop/valid": (n_step_history, n_tl_stop),  # bool
        "history/tl_stop/state": (n_step_history, n_tl_stop, 5),  # bool one_hot
        "history/tl_stop/pos": (n_step_history, n_tl_stop, 2),  # x,y
        "history/tl_stop/dir": (n_step_history, n_tl_stop, 2),  # dx,dy
    }

    batch = {
        "episode_idx": str(0),
        "scenario_id": scenario_id,
        "scenario_center": torch.Tensor([scenario_center]),
        "scenario_yaw": torch.Tensor(np.array([scenario_yaw])),
        "with_map": torch.Tensor([episode_with_map]),
    }
    for k, _size in tensor_size_test.items():
        batch[k] = torch.from_numpy(np.ascontiguousarray(episode_reduced[k]))[tf.newaxis]

    return batch

def save_video(batch, rollout, xodr_path, video_path):
    alpha_values = np.linspace(0.1, 0.8, 11) 
    scenario_center = batch["scenario_center"].unsqueeze(1)
    scenario_rot = torch_rad2rot(batch["scenario_yaw"])
    preds = torch_pos2global(rollout.preds[..., :2], scenario_center, scenario_rot)

    fig, ax_map_bg = plt.subplots(1, 1, figsize=(10, 10))
    ax_map_bg.set_aspect('equal')
    plot_xodr_map(ax_map_bg, xodr_path, draw_arrow=True)

    # 固定视角
    # vis_range = 200
    # ax_map_bg.set_xlim(preds[0, 0, 0, 0, 0] - vis_range, preds[0, 0, 0, 0, 0] + vis_range)
    # ax_map_bg.set_ylim(preds[0, 0, 0, 0, 1] - vis_range, preds[0, 0, 0, 0, 1] + vis_range)
    
    ax_objects = ax_map_bg.twinx()

    def update_objects(timestep):
        # # 跟随主车视角
        # vis_range = 100
        # ax_map_bg.set_xlim(preds[0, 0, 0, timestep, 0] - vis_range, preds[0, 0, 0, timestep, 0] + vis_range)
        # ax_map_bg.set_ylim(preds[0, 0, 0, timestep, 1] - vis_range, preds[0, 0, 0, timestep, 1] + vis_range)
        ax_objects.cla()
        ax_objects.axis('off')
        ax_objects.set_ylim(ax_map_bg.get_ylim())
        ax_objects.set_aspect('equal')

        for i in range(rollout.valid.shape[1]):
            if i == 0:
                color = 'red'
            else:
                color = 'teal'
            if (rollout.valid[0, i, 0, timestep]):
                x = preds[0, i, 0, max(0, timestep-10):timestep+1, 0]
                y = preds[0, i, 0, max(0, timestep-10):timestep+1, 1]
                ax_objects.scatter(x, y, s=8, marker='o', alpha=alpha_values, color=color)
                # ax_objects.text(x[-1], y[-1], str(int(batch['history/agent/object_id'][0, i])), ha='center', va='center', fontsize=8, color='black')
        return ax_objects,

    # 创建动画  
    ani = FuncAnimation(fig, update_objects, frames=np.arange(rollout.preds.shape[-2]), blit=False)  
    ani.save(video_path, writer='ffmpeg', fps=10)

def main():
    # scenario_id = "crossing_752_7_0"
    scenario_id = "0110follow103"
    scenario_id = "intersection_12_61_4"
    scenario_dir = f"../data/onsite_replay/{scenario_id}"
    for file_name in os.listdir(scenario_dir):
        if not file_name.startswith('.') and file_name.endswith('xodr'):
            xodr_path = os.path.join(scenario_dir, file_name)
        if not file_name.startswith('.') and file_name.endswith('xosc'):
            xosc_path = os.path.join(scenario_dir, file_name)

    cur_controller = IDM()

    CKPG_NAME = "2cti1q5z"
    CKPG_VER = "v41"
    # CKPG_NAME = "285yb3yb"
    # CKPG_VER = "v59"
    # 加载训练后的模型
    ckpt_path = f"../checkpoints/{CKPG_NAME}_{CKPG_VER}.ckpt"
    checkpoint_version = f"yangyh408/traffic_bots/{CKPG_NAME}:{CKPG_VER}"
    model = waymo_motion_module.WaymoMotion.load_from_checkpoint(
        ckpt_path, wb_artifact=checkpoint_version
    )

    batch = generate_batch_info(xodr_path, xosc_path, scenario_id)
    model.eval()
    with torch.no_grad():
        rollout = model.ego_test_loop_for_onsite(
            origin_batch=batch, 
            controller=cur_controller, 
            # goal = goal_expect,
            goal = None,
            sim_duration=90,
            k_futures = 1,
            print_info=True,
            rule_checker=False,
        )
    save_video(batch, rollout, xodr_path, f'../videos/{scenario_id}.mp4')

if __name__ == '__main__':
    main()