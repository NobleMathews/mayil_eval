import sys
from loguru import logger
from constants import swe_bench_tasks
import json
from pathlib import Path
from loguru import logger
from mayil import MayilIssue
from mayil.integrations.milvus import MilvusDB
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.tasks.process_issue import ProcessIssueTask
from mayil.ingestion.chunker import Chunk

import asyncio

MAYIL_VERSION = "v1"
COLLECTION_DIR = Path(f"./data/{MAYIL_VERSION}")

async def process_task(task_instance, testbed, ai_obj, db_obj) -> TaskReturnState:
    repo_path = (Path(testbed) / "repo").resolve()
    assert repo_path.exists()
    repo_name = f"testbed/{repo_path.parent.name}"
    fake_url = f"https://github.com/{repo_name}.git"
    title, body = task_instance["problem_statement"].split("\n", 1)
    _id = task_instance["instance_id"]
    issue_obj = MayilIssue(
        id=_id,
        repo_name=repo_name,
        title=title,
        body=body,
        state="closed",
        repo_link=fake_url,
    )
    process_issue_task = ProcessIssueTask(issue_obj=issue_obj)
    logger.info(f"Processing Issue {_id} in {repo_name}")
    result = await process_issue_task.run(db_obj=db_obj, ai_obj=ai_obj)
    logger.success(f"{_id} processed successfully")
    final_issue_obj = process_issue_task.issue_obj
    collected_data = final_issue_obj.to_json()

    def format_val(val):
        if isinstance(val, Chunk):
            return {
                "code": val.code,
                "filename": val.filename,
                "start_index": val.start_index,
                "end_index": val.end_index,
                "start_line": val.start_line,
                "end_line": val.end_line,
                "max_line": val.max_line,
                "git_instance": val.git_instance,
                "repo_name": val.repo_name,
                "sha": val.sha,
            }
        elif isinstance(val, list):
            return [format_val(item) for item in val]
        elif isinstance(val, dict):
            return {key: format_val(value) for key, value in val.items()}
        else:
            return val
    
    collected_data["mayil_collected_data"] = format_val(final_issue_obj.mayil_collected_data)
    return result, collected_data

async def main(total_processes, process_index):

    RedisClientSingleton()
    ai_obj = OpenAI()
    db_obj = MilvusDB()

    distributed_tasks = []
    with open(f"data/distributed_{swe_bench_tasks}.json", "r") as f:
        distributed_tasks = json.load(f)

    my_tasks = []
    for task_set in distributed_tasks:
        testbed = task_set["testbed"]
        task_instances = task_set["task_instances"]
        if total_processes == 1:
            my_tasks.append((task_instances[-1], testbed))
            break
        for i, task in enumerate(task_instances):
            output_path = COLLECTION_DIR / f"{collected_data['id']}.json"
            if output_path.exists():
                try:
                    with open(output_path, "r") as f:
                        collected_data = json.load(f)
                    if collected_data.get("mayil_collected_data", {}).get("status", "") == "completed":
                        continue
                except:
                    pass
            if i % total_processes == process_index:
                my_tasks.append(task, testbed)

    tasks_data = await asyncio.gather(*(process_task(task, testbed, ai_obj=ai_obj, db_obj=db_obj) for task, testbed in my_tasks))
    for j, task_data in enumerate(tasks_data):
        task_status, collected_data = task_data
        with open(COLLECTION_DIR / f"{collected_data['id']}.json", "w") as f:
            json.dump(collected_data, f, indent=4)
        logger.info(f"Process {process_index} | Task {j} status: {task_status}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        total_processes = int(sys.argv[1])
        process_index = int(sys.argv[2])
    else:
        logger.info(f"Usage: {sys.argv[0]} <total_processes> <process_index>")
        logger.warning("Running in debug mode")
        total_processes = 1
        process_index = 0

    if not COLLECTION_DIR.exists():
        COLLECTION_DIR.mkdir(parents=True)
    
    asyncio.run(main(total_processes, process_index))