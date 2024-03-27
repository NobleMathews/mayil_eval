from mayil.ingestion.vcs_handler import RepoParser
import json
from pathlib import Path
from loguru import logger
from mayil import MayilIssue
from mayil.integrations.milvus import MilvusDB
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.tasks.code_indexer import CodeIndexerTask
import asyncio
import sys
from constants import swe_bench_tasks

async def process_task_set(task_set, ai_obj, db_obj) -> TaskReturnState:
    task_instance = task_set["task_instances"][0]
    testbed = task_set["testbed"]
    # repo = task_instance["repo"]
    repo_path = (Path(testbed) / "repo").resolve()
    assert repo_path.exists()
    repo_name = f"testbed/{repo_path.parent.name}"
    fake_url = f"https://github.com/{repo_name}.git"
    immutable_repo_parser = RepoParser.from_folder(str(repo_path), fake_url)

    title, body = task_instance["problem_statement"].split("\n", 1)
    issue_obj = MayilIssue(
        id=task_instance["instance_id"],
        repo_name=repo_name,
        title=title,
        body=body,
        state="closed",
        repo_link=fake_url,
    )
    code_indexer_task = CodeIndexerTask(issue_obj=issue_obj)
    code_indexer_task.parser_from_file = immutable_repo_parser
    logger.info(f"Ingesting repo {repo_path}")
    result = await code_indexer_task.run(db_obj=db_obj, ai_obj=ai_obj)
    logger.success(f"{repo_name} ingested successfully")
    return result

async def main(total_processes, process_index):

    RedisClientSingleton()
    ai_obj = OpenAI()
    db_obj = MilvusDB()

    distributed_tasks = []
    with open(f"data/distributed_{swe_bench_tasks}.json", "r") as f:
        distributed_tasks = json.load(f)

    my_tasks = []
    for i, task_set in enumerate(distributed_tasks):
        if i % total_processes == process_index:
            if total_processes == 1:
                my_tasks.append(distributed_tasks[-2])
                break
            my_tasks.append(task_set)

    task_statuses = await asyncio.gather(*(process_task_set(task_set, ai_obj=ai_obj, db_obj=db_obj) for task_set in my_tasks))
    for j, task_status in enumerate(task_statuses):
        logger.info(f"Process {process_index} | Batch {i//10} | Task {j} status: {task_status}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        total_processes = int(sys.argv[1])
        process_index = int(sys.argv[2])
    else:
        print("Usage: python c_setup_mayil_ingestion.py <total_processes> <process_index>")
        total_processes = 1
        process_index = 0
    asyncio.run(main(total_processes, process_index))
