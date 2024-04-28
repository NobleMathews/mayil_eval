from loguru import logger
import json
import os
from pathlib import Path
from loguru import logger
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.agents.eval_agent import EvalAgent
from mayil.integrations.github import GitHubPublicAPI
from dotenv import load_dotenv
import asyncio

MAYIL_VERSION = "aftersell"
COLLECTION_DIR = Path(f"./data/GroundTruthScoring/{MAYIL_VERSION}")
current_cost = 0

async def process_task(task_instance, ai_obj) -> TaskReturnState:
    eval_agent_task = EvalAgent()
    logger.info(f"Processing {task_instance[0]}")
    try:
        collected_data = await eval_agent_task.run(mayil_response=task_instance[1], diff=task_instance[2], issue_body=task_instance[3], ai_obj=ai_obj)
    except Exception as e:
        logger.error(f"Error processing {task_instance[0]}: {e}")
        return {}
    logger.success(f"{task_instance[0]} processed successfully")
    
    collected_data["id"] = task_instance[0]
    return collected_data

async def main():
    global current_cost
    load_dotenv(".env")
    load_dotenv(".env.local")

    GITHUB_TOKEN = os.getenv("TESTING_GITHUB_TOKEN")

    git_api = GitHubPublicAPI(
        token=GITHUB_TOKEN
    )

    RedisClientSingleton()
    ai_obj = OpenAI()
    # db_obj = MilvusDB()

    # with open("./data/test.json", "r") as f:
    #     ground_truth = json.load(f)

    test_cases = {
        "testbed/aftersell":{
            # "BEAM-2996",
            # https://github.com/ROKT/aftersell/pull/1083
            # "BEAM-3120",
            # https://github.com/ROKT/aftersell/pull/1129
            "BEAM-3220": 1120,
            "BEAM-3449": 1119,
            "BEAM-3375": 1105,
            "BEAM-3118": 1103,
            "BEAM-3125": 1121,
            "BEAM-3157": 1116,
            "BEAM-3254": 1100,
            "BEAM-3123": 1102,
            "BEAM-3124": 1107,
            "BEAM-3113": 1108,
            "BEAM-3064": 1122,
        },
        "testbed/UpCart-2.0":{
            "BEAM-2284": 420,
            "BEAM-2750": 428,
            "BEAM-2762": 429,
            "BEAM-2793": 433,
        }
    }

    ground_truth = []
    for repo_name, issues in test_cases.items():
        for issue_id, item_id in issues.items():
            issue_details = {
                "instance_id": issue_id,
                "problem_statement": None,
                "patch": git_api.get_pr_patch("beam-commerce/"+repo_name.split("/")[-1], int(item_id)),
                "repo_name": repo_name,
            }
            ground_truth.append(issue_details)
            

    task_instances = []
    for issue_details in ground_truth:
        path_to_ground_truth = f"./data/{MAYIL_VERSION}/{issue_details['instance_id']}.json"
        if not Path(path_to_ground_truth).exists():
            continue
        with open(path_to_ground_truth, "r") as f:
            generated_details = json.load(f)
        ground_truth_diff = issue_details["patch"]
        problem_statement = issue_details["problem_statement"]
        if problem_statement is None:
            problem_statement = generated_details["title"] + "\n\n" + generated_details["body"]
        mayil_response = generated_details["mayil_collected_data"]["result"]
        if not mayil_response:
            continue
        assert ground_truth_diff
        assert problem_statement
        assert mayil_response
        task_instances.append((issue_details['instance_id'],mayil_response, ground_truth_diff, problem_statement))

    batch_size = 50
    for task_batch in [task_instances[i:i+batch_size] for i in range(0, len(task_instances), batch_size)]: 
        if current_cost > 500:
            logger.error(f"Cost exceeded 500")
            break
        filtered_task_batch = []
        for task in task_batch:
            output_path = COLLECTION_DIR / f"{task[0]}.json"
            if output_path.exists():
                try:
                    with open(output_path, "r") as f:
                        collected_data = json.load(f)
                        current_cost += collected_data["cost"]
                        continue
                except:
                    pass
            filtered_task_batch.append(task)
        collected_data = await asyncio.gather(*(process_task(task, ai_obj=ai_obj) for task in filtered_task_batch))
        for task_data in collected_data:
            current_cost += task_data["cost"]
            with open(COLLECTION_DIR / f"{task_data['id']}.json", "w") as f:
                json.dump(task_data, f, indent=4)


if __name__ == "__main__":

    if not COLLECTION_DIR.exists():
        COLLECTION_DIR.mkdir(parents=True)
    
    asyncio.run(main())