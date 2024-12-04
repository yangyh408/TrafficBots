import torch
import tensorflow as tf
import numpy as np 
import pickle
from waymo_open_dataset.protos import scenario_pb2

from pack_h5_womd import collate_map_features, collate_tl_features
import utils.pack_h5 as pack_utils
import pl_modules.waymo_motion as waymo_motion_module

from controller.manual import ManualController
from controller.idm import IDM

SCENE_IDX = 0
CKPG_NAME = "2cti1q5z"
CKPG_VER = "v41"
# HOME_DIR = "/home/yangyh408/codes/DataDriven_BehaviourModels/deep_learning_model/v1.0"
HOME_DIR = "/home/gz0779601/codes/DataDriven_BehaviourModels/deep_learning_model/v1.0"

# -----------------------------------------------------------------------------------------------------

# id [agent_num]
agent_id = [369, 370, 371, 372, 556]

# type [agent_num]
# [TYPE_VEHICLE=0, TYPE_PEDESTRIAN=1, TYPE_CYCLIST=2]
agent_type = [0, 0, 0, 0, 0]

# role [agent_num, 3]
# 0 - 是否为主车
# 1 - 是否为objects_of_interest（可能对研究训练有用的行为的对象）
# 2 - 是否为tracks_to_predict（必须预测的对象，仅在训练集和验证集中提供）
agent_role = [
    [False, False, False],
    [False, False, True],
    [False, False, False],
    [False, False, False],
    [True, False, False]
]

# states [agent_num, frames, 10]
# [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
agent_states = [
    [[8395.5048828125, 7249.99609375, -14.010635375976562, 4.756037712097168, 2.0462944507598877, 1.6023485660552979, 1.5822010040283203, -0.3125, 19.326171875, True], [8395.4736328125, 7251.9287109375, -14.000516163384034, 4.76189661026001, 2.0593159198760986, 1.6107460260391235, 1.585012435913086, -0.3125, 19.326171875, True], [8395.431640625, 7253.8505859375, -13.998344045855672, 4.739120006561279, 2.0614328384399414, 1.619963526725769, 1.583803415298462, -0.419921875, 19.21875, True], [8395.4013671875, 7255.80859375, -14.015545551811453, 4.771324634552002, 2.0703282356262207, 1.6340086460113525, 1.5805013179779053, -0.302734375, 19.580078125, True], [8395.3798828125, 7257.7041015625, -14.036089769789383, 4.775165557861328, 2.0761101245880127, 1.6181843280792236, 1.578407645225525, -0.21484375, 18.955078125, True], [8395.35546875, 7259.6025390625, -14.01985303451912, 4.774107456207275, 2.0866475105285645, 1.652348279953003, 1.588748812675476, -0.244140625, 18.984375, True], [8395.349609375, 7261.51171875, -14.038302999529211, 4.753843784332275, 2.082709312438965, 1.6332414150238037, 1.5885826349258423, -0.05859375, 19.091796875, True], [8395.3271484375, 7263.4248046875, -14.058067530551314, 4.747679233551025, 2.087566614151001, 1.6209542751312256, 1.5893115997314453, -0.224609375, 19.130859375, True], [8395.2841796875, 7265.3115234375, -14.074770327401692, 4.710694789886475, 2.0733964443206787, 1.6461181640625, 1.5896310806274414, -0.4296875, 18.8671875, True], [8395.2529296875, 7267.17919921875, -14.071941190266063, 4.6964521408081055, 2.059957265853882, 1.6449100971221924, 1.585894227027893, -0.3125, 18.6767578125, True], [8395.2197265625, 7269.10791015625, -14.083323505194313, 4.745246410369873, 2.0748298168182373, 1.6331822872161865, 1.590860366821289, -0.33203125, 19.287109375, True]],
    [[8396.10546875, 7154.03076171875, -13.450148582458496, 5.133871555328369, 2.203359365463257, 1.6165982484817505, 1.5685367584228516, 0.13671875, 21.376953125, True], [8396.119140625, 7156.16845703125, -13.444271313225343, 5.091489315032959, 2.2021641731262207, 1.6125479936599731, 1.563308596611023, 0.13671875, 21.376953125, True], [8396.1376953125, 7158.24462890625, -13.432815176226766, 5.096654415130615, 2.1918182373046875, 1.587924838066101, 1.567080020904541, 0.185546875, 20.76171875, True], [8396.123046875, 7160.42724609375, -13.464530651603933, 5.138164043426514, 2.2315266132354736, 1.592545747756958, 1.567673683166504, -0.146484375, 21.826171875, True], [8396.12109375, 7162.57958984375, -13.448625437208817, 5.110944747924805, 2.2167716026306152, 1.5978059768676758, 1.5649570226669312, -0.01953125, 21.5234375, True], [8396.10546875, 7164.7529296875, -13.49083177139656, 5.104405879974365, 2.2150659561157227, 1.5922120809555054, 1.564302682876587, -0.15625, 21.7333984375, True], [8396.1015625, 7166.8642578125, -13.483409505876867, 5.084568023681641, 2.212529182434082, 1.6014478206634521, 1.580500841140747, -0.0390625, 21.11328125, True], [8396.0947265625, 7169.013671875, -13.516496867099166, 5.117166042327881, 2.199558973312378, 1.6021273136138916, 1.5702431201934814, -0.068359375, 21.494140625, True], [8396.0908203125, 7171.1484375, -13.504861231637532, 5.035835266113281, 2.1944468021392822, 1.5906894207000732, 1.5694551467895508, -0.0390625, 21.34765625, True], [8396.0830078125, 7173.26025390625, -13.515580945515087, 5.100960731506348, 2.2359704971313477, 1.6081032752990723, 1.56867253780365, -0.078125, 21.1181640625, True], [8396.06640625, 7175.42333984375, -13.496037509710915, 5.0914740562438965, 2.2275919914245605, 1.6015872955322266, 1.5792957544326782, -0.166015625, 21.630859375, True]],
    [[8392.181640625, 7235.14306640625, -13.830416679382324, 4.693872451782227, 2.0756969451904297, 1.5766584873199463, 1.5732872486114502, -0.1171875, 18.7109375, True], [8392.169921875, 7237.01416015625, -13.838374363457277, 4.681710720062256, 2.0667858123779297, 1.5723609924316406, 1.5711127519607544, -0.1171875, 18.7109375, True], [8392.177734375, 7238.86328125, -13.843545538165243, 4.7035932540893555, 2.078188180923462, 1.5730082988739014, 1.5785671472549438, 0.078125, 18.4912109375, True], [8392.171875, 7240.6875, -13.843564693962332, 4.66843843460083, 2.0769972801208496, 1.5871800184249878, 1.5767985582351685, -0.05859375, 18.2421875, True], [8392.16796875, 7242.51025390625, -13.850698343702957, 4.6832075119018555, 2.082688808441162, 1.5789870023727417, 1.5782173871994019, -0.0390625, 18.2275390625, True], [8392.1611328125, 7244.34619140625, -13.853792586826248, 4.724412441253662, 2.0911755561828613, 1.5947036743164062, 1.5796922445297241, -0.068359375, 18.359375, True], [8392.15625, 7246.17822265625, -13.871447187456457, 4.7202839851379395, 2.0909836292266846, 1.591770887374878, 1.5785704851150513, -0.048828125, 18.3203125, True], [8392.1435546875, 7247.990234375, -13.89080068484819, 4.704876899719238, 2.0847349166870117, 1.5879217386245728, 1.5807387828826904, -0.126953125, 18.1201171875, True], [8392.130859375, 7249.81494140625, -13.903387423348958, 4.704655647277832, 2.0848331451416016, 1.5925785303115845, 1.5814419984817505, -0.126953125, 18.2470703125, True], [8392.1220703125, 7251.6357421875, -13.920621686481883, 4.697376251220703, 2.0858945846557617, 1.5977681875228882, 1.5784320831298828, -0.087890625, 18.2080078125, True], [8392.1162109375, 7253.4599609375, -13.920165088446266, 4.715240955352783, 2.0954883098602295, 1.6111465692520142, 1.5782873630523682, -0.05859375, 18.2421875, True]],
    [[8392.6533203125, 7174.14501953125, -13.4308500289917, 4.81757116317749, 2.0840933322906494, 1.5348986387252808, 1.5724315643310547, 0.078125, 17.548828125, True], [8392.6611328125, 7175.89990234375, -13.443473087822511, 4.858018398284912, 2.1152150630950928, 1.5546501874923706, 1.5757981538772583, 0.078125, 17.548828125, True], [8392.654296875, 7177.6904296875, -13.443202596881063, 4.834859371185303, 2.1077427864074707, 1.5157395601272583, 1.5744959115982056, -0.068359375, 17.9052734375, True], [8392.640625, 7179.419921875, -13.450582211052176, 4.84534215927124, 2.092646360397339, 1.5199711322784424, 1.5743408203125, -0.13671875, 17.294921875, True], [8392.6220703125, 7181.13818359375, -13.463391176649735, 4.890226364135742, 2.113778829574585, 1.5292786359786987, 1.5733261108398438, -0.185546875, 17.1826171875, True], [8392.603515625, 7182.9052734375, -13.475655951999588, 4.865891456604004, 2.111224412918091, 1.5349719524383545, 1.5736961364746094, -0.185546875, 17.6708984375, True], [8392.576171875, 7184.65380859375, -13.487830740007727, 4.859665870666504, 2.1141223907470703, 1.5380562543869019, 1.582428216934204, -0.2734375, 17.4853515625, True], [8392.5615234375, 7186.390625, -13.49995252505815, 4.846556663513184, 2.100978136062622, 1.5447444915771484, 1.5824164152145386, -0.146484375, 17.3681640625, True], [8392.5595703125, 7188.08642578125, -13.514174815011556, 4.886437892913818, 2.120779514312744, 1.5530259609222412, 1.5744459629058838, -0.01953125, 16.9580078125, True], [8392.533203125, 7189.85693359375, -13.53062897255366, 4.878307819366455, 2.1198782920837402, 1.5357599258422852, 1.575266718864441, -0.263671875, 17.705078125, True], [8392.513671875, 7191.58544921875, -13.522732761175758, 4.912601470947266, 2.1399970054626465, 1.5449631214141846, 1.5798118114471436, -0.1953125, 17.28515625, True]],
    [[8392.410571507779, 7202.185541983308, -13.144874041140794, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5756219625473022, -0.08884891867637634, 18.20938491821289, True], [8392.401683171454, 7204.007186346924, -13.15710868576702, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5759450197219849, -0.08954513072967529, 18.22984504699707, True], [8392.392658314086, 7205.832362924224, -13.165779654191589, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.576280951499939, -0.09516873210668564, 18.27100944519043, True], [8392.382652888338, 7207.6607696605215, -13.175107580238956, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5764985084533691, -0.10137411952018738, 18.307140350341797, True], [8392.37238727034, 7209.493098445325, -13.18746239938947, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5765782594680786, -0.09894900023937225, 18.338048934936523, True], [8392.36286564337, 7211.327881404639, -13.197656493381832, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5766102075576782, -0.09478527307510376, 18.3569393157959, True], [8392.35343316377, 7213.163918421533, -13.205582912724745, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5766372680664062, -0.09327118843793869, 18.36125946044922, True], [8392.344211114769, 7215.000191170081, -13.215478621050638, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5766922235488892, -0.09516789019107819, 18.364782333374023, True], [8392.334398732912, 7216.837037891057, -13.228209585335383, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5766382217407227, -0.0965147539973259, 18.374237060546875, True], [8392.324907212298, 7218.675219194855, -13.238801678645427, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5766737461090088, -0.09474021941423416, 18.38309669494629, True], [8392.315450503396, 7220.5136930799345, -13.248955714300955, 5.285999774932861, 2.3320000171661377, 2.3299999237060547, 1.5767649412155151, -0.0999351516366005, 18.378984451293945, True]]
]

goal_expect = [239, 269, 269, 269, 269]

# -----------------------------------------------------------------------------------------------------

N_PL_TYPE = 11
DIM_VEH_LANES = [0, 1, 2]
DIM_CYC_LANES = [3]
DIM_PED_LANES = [4]

N_TL_STATE = 5

N_PL_MAX = 3000
N_TL_MAX = 40
N_AGENT_MAX = 1300

N_PL = 1024
N_PL_NODE = 20
N_TL = 100
N_AGENT = 64
N_AGENT_NO_SIM = 256

THRESH_MAP = 500  # ! 120
THRESH_AGENT = 120

N_STEP = 91
N_STEP_HISTORY = 11
STEP_CURRENT = 10

pack_all = False
pack_history = True

# -----------------------------------------------------------------------------------------------------

# 从数据集加载场景
with open(rf"../data/h5_wosac/val_scenarios/{SCENE_IDX}.pickle", "rb") as handle:
    scenario = scenario_pb2.Scenario.FromString(bytes.fromhex(pickle.load(handle).hex()))

# 将场景打包成batch信息
mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(scenario.map_features)
tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(scenario.dynamic_map_states)

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

tensor_size_test = {
    # object_id for waymo metrics
    "history/agent/object_id": (N_AGENT,),
    "history/agent_no_sim/object_id": (N_AGENT_NO_SIM,),
    # agent_sim
    "history/agent/valid": (N_STEP_HISTORY, N_AGENT),  # bool,
    "history/agent/pos": (N_STEP_HISTORY, N_AGENT, 2),  # float32
    "history/agent/z": (N_STEP_HISTORY, N_AGENT, 1),  # float32
    "history/agent/vel": (N_STEP_HISTORY, N_AGENT, 2),  # float32, v_x, v_y
    "history/agent/spd": (N_STEP_HISTORY, N_AGENT, 1),  # norm of vel, signed using yaw_bbox and vel_xy
    "history/agent/acc": (N_STEP_HISTORY, N_AGENT, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
    "history/agent/yaw_bbox": (N_STEP_HISTORY, N_AGENT, 1),  # float32, yaw of the bbox heading
    "history/agent/yaw_rate": (N_STEP_HISTORY, N_AGENT, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
    "history/agent/type": (N_AGENT, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
    "history/agent/role": (N_AGENT, 3),  # bool [sdc=0, interest=1, predict=2]
    "history/agent/size": (N_AGENT, 3),  # float32: [length, width, height]
    "history/agent_no_sim/valid": (N_STEP_HISTORY, N_AGENT_NO_SIM),
    "history/agent_no_sim/pos": (N_STEP_HISTORY, N_AGENT_NO_SIM, 2),
    "history/agent_no_sim/z": (N_STEP_HISTORY, N_AGENT_NO_SIM, 1),
    "history/agent_no_sim/vel": (N_STEP_HISTORY, N_AGENT_NO_SIM, 2),
    "history/agent_no_sim/spd": (N_STEP_HISTORY, N_AGENT_NO_SIM, 1),
    "history/agent_no_sim/yaw_bbox": (N_STEP_HISTORY, N_AGENT_NO_SIM, 1),
    "history/agent_no_sim/type": (N_AGENT_NO_SIM, 3),
    "history/agent_no_sim/size": (N_AGENT_NO_SIM, 3),
    # map
    "map/valid": (N_PL, N_PL_NODE),  # bool
    "map/type": (N_PL, 11),  # bool one_hot
    "map/pos": (N_PL, N_PL_NODE, 2),  # float32
    "map/dir": (N_PL, N_PL_NODE, 2),  # float32
    "map/boundary": (4,),  # xmin, xmax, ymin, ymax
    # traffic_light
    "history/tl_lane/valid": (N_STEP_HISTORY, N_TL),  # bool
    "history/tl_lane/state": (N_STEP_HISTORY, N_TL, 5),  # bool one_hot
    "history/tl_lane/idx": (N_STEP_HISTORY, N_TL),  # int, -1 means not valid
    "history/tl_stop/valid": (N_STEP_HISTORY, N_TL_MAX),  # bool
    "history/tl_stop/state": (N_STEP_HISTORY, N_TL_MAX, 5),  # bool one_hot
    "history/tl_stop/pos": (N_STEP_HISTORY, N_TL_MAX, 2),  # x,y
    "history/tl_stop/dir": (N_STEP_HISTORY, N_TL_MAX, 2),  # dx,dy
}

batch = {
    "episode_idx": str(0),
    "scenario_id": scenario.scenario_id,
    "scenario_center": torch.Tensor([scenario_center]),
    "scenario_yaw": torch.Tensor(np.array([scenario_yaw])),
    "with_map": torch.Tensor([episode_with_map]),
}
for k, _size in tensor_size_test.items():
    batch[k] = torch.from_numpy(np.ascontiguousarray(episode_reduced[k]))[tf.newaxis]

# 加载规控器
cur_controller = IDM()

# 加载训练后的模型
ckpt_path = f"../checkpoints/{CKPG_NAME}_{CKPG_VER}.ckpt"
checkpoint_version = f"yangyh408/traffic_bots/{CKPG_NAME}:{CKPG_VER}"
model = waymo_motion_module.WaymoMotion.load_from_checkpoint(
    ckpt_path, wb_artifact=checkpoint_version
)

# 运行闭环测试
model.eval()
with torch.no_grad():
    rollout = model.ego_test_loop(
        origin_batch=batch, 
        scenario=scenario, 
        controller=cur_controller, 
        goal = goal_expect,
        sim_duration=90,
        k_futures = 1,
        print_info=True,
        rule_checker=False,
        visualize=True,
        video_path=f"videos/test_{batch['episode_idx'][0]}.mp4"
    )