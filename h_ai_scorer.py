from loguru import logger
import json
from pathlib import Path
from loguru import logger
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.agents.eval_agent import EvalAgent

import asyncio

MAYIL_VERSION = "v1"
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

    RedisClientSingleton()
    ai_obj = OpenAI()
    # db_obj = MilvusDB()

    with open("./data/test.json", "r") as f:
        ground_truth = json.load(f)

    task_instances = []
    for issue_details in ground_truth:
        with open(f"./data/v1/{issue_details['instance_id']}.json", "r") as f:
            generated_details = json.load(f)
        ground_truth_diff = issue_details["patch"]
        problem_statement = issue_details["problem_statement"]
        mayil_response = generated_details["mayil_response"]
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