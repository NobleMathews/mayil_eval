import os
import re
from loguru import logger
from collections import defaultdict
from constants import swe_bench_tasks
import json
from mayil.ingestion.vcs_handler import RepoParser
from pathlib import Path
from loguru import logger
from mayil import MayilIssue
from mayil.integrations.milvus import MilvusDB
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.tasks.code_indexer import CodeIndexerTask
import asyncio

RedisClientSingleton()
ai_obj = OpenAI()
db_obj = MilvusDB()

error_files = defaultdict(list)
for file in os.listdir('./logs'):
    if file.endswith('.log'):
        with open(os.path.join('./logs', file), 'r') as f:
            for line in f:
                if '| ERROR' in line:
                    match = re.search(r'Error indexing file: (.*) in repo: (.*) but continuing', line)
                    if match:
                        error_files[match.group(2)].append(match.group(1))
                        logger.info(f"Found missing file {match.group(1)} in repo {match.group(2)}")
                    else:
                        raise Exception(f"Error message not in expected format: {line}")

distributed_tasks = []
with open(f"data/distributed_{swe_bench_tasks}.json", "r") as f:
    distributed_tasks = json.load(f)

my_tasks = []
for task_set in distributed_tasks:
    identifier = f"testbed/{task_set['venv']}"
    if  identifier in error_files:
        task_set['files'] = error_files[identifier]
        my_tasks.append(task_set)

async def process_task_set(task_set) -> TaskReturnState:
    missing_files = task_set['files']
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
    code_indexer_task.files_to_add = set(missing_files)
    logger.info(f"Ingesting repo {repo_path}: Adding {len(missing_files)} files")
    result = await code_indexer_task.run(db_obj=db_obj, ai_obj=ai_obj)
    logger.success(f"{repo_name} ingested successfully")
    return result

async def main():
    task_statuses = await asyncio.gather(*(process_task_set(task_set) for task_set in my_tasks))
    for j, task_status in enumerate(task_statuses):
        logger.info(f"Task {j} status: {task_status}")


if __name__ == "__main__":
    asyncio.run(main())