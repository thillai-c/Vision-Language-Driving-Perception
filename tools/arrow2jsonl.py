import argparse
import json
import os 
from tqdm import tqdm
from datasets import Dataset, Value, arrow_dataset

input_ids = f'''
        You are an expert in Driving Scene analysing. You will be shown the picture of the front perspective view of the ego car. Your task here is to describe the future trajectory prediction of the ego vehicle using immmeta actions.

        ### Meta-actions:
        Based on the point above, you should grasp the position change of the ego car. You can use these meta-actions to describe the immediate action of the ego vehicle. These actions include:
            - Speed-control actions: speed up, slow down, stop, wait
            - Turning actions: turn left, turn right, turn around
            - Lane-control actions: change lane, shift slightly to the left or right

        your output format should be the following structure:
        $$Meta Action$$ "Scene Description": "$$the description of the surrounding environment$$"
    '''

# Meta actions defined in VLM
meta_actions = {
    # speed control
    "speed up", 
    "slow down", 
    "slow down rapidly", 
    "go straight slowly", 
    "go straight at a constant speed", 
    "stop", 
    "wait", 
    "reverse", 
    # turn action
    "turn left", 
    "turn right", 
    "turn around", 
    # lane change
    "change lane to the left", 
    "change lane to the right", 
    "shift slightly to the left", 
    "shift slightly to the right", 
}


# Handle exceptions in LLM annotation
action_category_mapping = {
    # Speed Control
    "Speed up": "speed up",
    "accelerate": "speed up",
    "accelerate and continue left turn": "speed up",
    "accelerate slightly": "speed up",
    "adjust speed": "speed up",  # General speed control
    "continue at constant speed": "go straight at a constant speed",
    "continue at current speed": "go straight at a constant speed",
    "maintain constant speed": "go straight at a constant speed",
    "maintain speed": "go straight at a constant speed",
    "maintain speed and lane": "go straight at a constant speed",
    "maintain speed and shift slightly to the left": "shift slightly to the left",
    "maintain speed and shift slightly to the right": "shift slightly to the right",
    "slightly slow down": "slow down",
    "speed up and shift slightly to the left": "shift slightly to the left",
    "speed up and shift slightly to the right": "shift slightly to the right",
    "speed up slightly": "speed up", 
    "maintain current speed": "go straight at a constant speed",

    # Turn Action
    "continue turning left": "turn left",
    "slight turn to the right": "turn right",
    "slightly turn left": "turn left",
    "slightly turn right": "turn right",
    "turn more left": "turn left",
    "turn more sharply right": "turn right",
    "turn right slightly": "turn right",
    "turn sharp right": "turn right",
    "turn sharply left": "turn left",
    "turn sharply right": "turn right",
    "turn sharply to the right": "turn right",
    "turn slight left": "turn left",
    "turn slight right": "turn right",
    "turn slightly left": "turn left",
    "turn slightly right": "turn right",
    "turn slightly to the left": "turn left",
    "turn slightly to the right": "turn right",
    "turn to the right": "turn right",

    # Lane Change
    "adjust to the center of the lane": "go straight at a constant speed",
    # "change lane": "change lane to the right",  # need extra handling
    "change lane slightly to the right": "change lane to the right",
    "maintain lane": "go straight at a constant speed",
    "maintain lane and speed": "go straight at a constant speed",
    "maintain lane position": "go straight at a constant speed",
    "maintain lane with slight adjustments": "go straight at a constant speed",
    "shift more to the left": "shift slightly to the left",
    "shift significantly to the left": "shift slightly to the left",
    "shift significantly to the right": "shift slightly to the right",
    "Shift slightly to the left": "shift slightly to the left",
    "shift slightly left": "shift slightly to the left",
    "shift slightly right": "shift slightly to the right",
    "shift to the left": "shift slightly to the left",
    "shift to the right": "shift slightly to the right",
    "shift to the right lane": "change lane to the right",
    "slight left shift": "shift slightly to the left",
    "slight right shift": "shift slightly to the right",
    "slight shift right": "shift slightly to the right",
    "slight shift to the right": "shift slightly to the right",
    "slightly adjust to the left": "shift slightly to the left",
    "slightly shift left": "shift slightly to the left",
    "slightly shift right": "shift slightly to the right",
    "slightly shift to the left": "shift slightly to the left",
    "slightly shift to the right": "shift slightly to the right",

    # Continue/Position (Closest Meta Actions)
    "adjust course": "go straight at a constant speed",
    "continue forward": "go straight at a constant speed",
    "continue straight": "go straight at a constant speed",
    "go straight": "go straight at a constant speed",
    "maintain current lane": "go straight at a constant speed",
    "maintain current position": "wait",
    "maintain position": "wait",
    "maintain straight": "go straight at a constant speed",
    "move forward": "go straight at a constant speed",
    "move forward slightly to the right": "shift slightly to the right",
    "move straight": "go straight at a constant speed",
    "stabilize": "go straight at a constant speed",
    "stay in lane": "go straight at a constant speed"
}

def mapping_action(meta_action, subject):
    if meta_action not in meta_actions:
        if meta_action in action_category_mapping.keys():
            meta_action = action_category_mapping[meta_action]
        elif meta_action == "change lane":
            if "left" in subject:
                meta_action = "change lane to the left"
            elif "right" in subject:
                meta_action = "change lane to the right"
            else:
                print("Failed to map action: ", meta_action, " Use default action: go straight at a constant speed")
                meta_action = "go straight at a constant speed"
        else:
            print(f"Action {meta_action} not in meta_actions or action_category_mapping, use default action: go straight at a constant speed")
            meta_action = "go straight at a constant speed"
    return meta_action

def load_dataset(root, split='train', dataset_scale=1, select=False, debug = False):
    datasets = []
    index_root_folders = root
    indices = os.listdir(index_root_folders)
    # ['train-index_vegas3', 'train-index_singapore', 'train-index_pittsburgh', 'train-index_vegas2', 'train-index_vegas4', 'train-index_vegas5', 'train-index_boston', 'train-index_vegas1', 'train-index_vegas6']
    if debug:
        print(indices)
        indices = indices[1:2]
    for index in indices:
        index_path = os.path.join(index_root_folders, index)
        if os.path.isdir(index_path):
            # load training dataset
            dataset = Dataset.load_from_disk(index_path)
            if dataset is not None:
                datasets.append(dataset)
        else:
            continue
    # For nuplan dataset directory structure, each split obtains multi cities directories, so concat is required;
    # But for waymo dataset, index directory is just the datset, so load directory directly to build dataset. 
    if len(datasets) > 0: 
        dataset = arrow_dataset._concatenate_map_style_datasets(datasets)
        for each in datasets:
            each.cleanup_cache_files()
    else: 
        dataset = Dataset.load_from_disk(index_root_folders)

    # add split column
    dataset.features.update({'split': Value('string')})
    try:
        dataset = dataset.add_column(name='split', column=[split] * len(dataset))
    except:
        pass
    
    dataset.features.update({'input_ids': Value('string')})
    try:
        dataset = dataset.add_column(name='input_ids', column=[input_ids] * len(dataset))
    except:
        pass

    dataset.set_format(type='torch')

    if select:
        samples = int(len(dataset) * float(dataset_scale))
        dataset = dataset.select(range(samples))

    return dataset

def main(args):
    assert args.data_root is not None and args.output_file is not None
    for dir in os.listdir(args.data_root):
        if "singapore" in dir: continue
        # if args.split in dir:
        dataset = load_dataset(os.path.join(args.data_root, dir), args.split, 1, True)

        with open(args.output_file, "a+") as file:
            for i, d in tqdm(enumerate(dataset)):
                meta_action = d["hierarchical_planning"].split("Action\": \"")[1].split("\"")[0]
                subject = d["hierarchical_planning"].split("Action\": \"")[1].split("\"")[4]
                meta_action = mapping_action(meta_action, subject)
                scene = d["scene_analysis"].split("Scene_Summary\": \"")[-1].split("\"")[0]
                path_parts = d["image_path"].split('/')
                short_path = '/'.join(path_parts[-3:])
                d["image_path"] = short_path
                data = {"id": i, "image": d["image_path"], "width": 1920, "height": 1080, 
                        "conversations": [{"from": "human", "value": d["input_ids"]}, 
                                            {"from": "gpt", "value": "$$" + meta_action + "$$" + scene}]}
                file.write(json.dumps(data) + "\n")
        
        print("Convert", len(dataset), dir, "samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    
    main(args)
'''
python ./vlm/tools/arrow2jsonl.py \
    --data_root ./data/dataset/vlm_ann_arrow \
    --output_file ./data/dataset/vlm_ann_train.jsonl \
    --split train
'''