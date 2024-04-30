from loguru import logger
from constants import swe_bench_tasks
import json
import os
from pathlib import Path
from loguru import logger
from mayil import MayilIssue
from mayil.integrations.milvus import MilvusDB
from mayil.integrations.redis import RedisClientSingleton
from mayil.integrations.openai import OpenAI
from mayil.tasks import TaskReturnState
from mayil.tasks.issue_completion import IssueCompletionTask
from mayil import ExternalAssetsType
from mayil.tasks.process_issue import ProcessIssueTask
from mayil.chunk import Chunk
from mayil.integrations.repostore import RepoStore
from mayil.ingestion.repo_parser import RepoParser
from mayil.integrations.linear import LinearBot

import asyncio

MAYIL_VERSION = "aftersell"
COLLECTION_DIR = Path(f"./data/{MAYIL_VERSION}")
current_cost = 0

async def process_task(task_instance, testbed, ai_obj, db_obj) -> TaskReturnState:
    repo_path = (Path(testbed) / "repo").resolve()
    if not repo_path.exists():
        repo_path = Path(testbed).resolve()
        assert repo_path.exists()
        repo_name = f"testbed/{repo_path.name}"
    else:
        repo_name = f"testbed/{repo_path.parent.name}"
    fake_url = f"https://github.com/{repo_name}.git"
    immutable_repo_parser = RepoParser.from_folder(repo_path, fake_url)
    RepoStore.repoparser_lookup = {}
    RepoStore.repoparser_lookup[fake_url] = immutable_repo_parser
    _id = task_instance["instance_id"]
    if "problem_statement" in task_instance:
        title, body = task_instance["problem_statement"].split("\n", 1)
        issue_obj = MayilIssue(
            id=_id,
            repo_name=repo_name,
            title=title,
            body=body,
            state="closed",
            repo_link=fake_url,
        )
    else:
        # issue_and_parents = task_instance["issue_and_parents"]
        
        issue_obj = MayilIssue(
            repo_name=repo_name,
            state="closed",
            repo_link=fake_url,
            **LinearBot().get_issue_details(
                issue_id=_id,
                include_comments=False
            ),
        )

        for linked_issue_id in issue_obj.linked_issue_ids:
            linked_issue_obj = MayilIssue(
                repo_name=repo_name,
                state="open",
                repo_link=fake_url,
                **LinearBot().get_issue_details(
                    issue_id=linked_issue_id,
                    include_comments=False
                ),
            )
            assert linked_issue_obj.linked_issue_ids == []
            completion_task = IssueCompletionTask(linked_issue_obj)
            completion_task.disallowed_asset_types = [ExternalAssetsType.GITHUB_PR, ExternalAssetsType.DEVELOPER_COMMENTS]
            await completion_task.run(ai_obj=ai_obj)
            await ProcessIssueTask(linked_issue_obj).run(ai_obj=ai_obj)


        issue_completion_task = IssueCompletionTask(issue_obj)
        issue_completion_task.disallowed_asset_types = [ExternalAssetsType.GITHUB_PR, ExternalAssetsType.DEVELOPER_COMMENTS]
        logger.info(f"Completing Issue {_id} in {repo_name}")
        result = await issue_completion_task.run(ai_obj=ai_obj)
        issue_obj = issue_completion_task.issue_obj
    process_issue_task = ProcessIssueTask(issue_obj=issue_obj)
    logger.info(f"Processing Issue {_id} in {repo_name}")
    result = await process_issue_task.run(ai_obj=ai_obj)
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
# total_processes, process_index
async def main(distributed_tasks):
    global current_cost

    ai_obj = OpenAI()
    db_obj = MilvusDB()

    if not distributed_tasks:
        with open(f"data/distributed_{swe_bench_tasks}.json", "r") as f:
            distributed_tasks = json.load(f)

    # my_tasks = []
    for task_set in distributed_tasks:
        testbed = task_set["testbed"]
        task_instances = task_set["task_instances"]
        # if total_processes == 1:
        #     my_tasks.append((task_instances[-1], testbed))
        #     break
        batch_size = 25
        for task_batch in [task_instances[i:i+batch_size] for i in range(0, len(task_instances), batch_size)]: 
            if current_cost > 500:
                logger.error(f"Cost exceeded 200")
                break
            filtered_task_batch = []
            for task in task_batch:
                output_path = COLLECTION_DIR / f"{task['instance_id']}.json"
                if output_path.exists():
                    try:
                        with open(output_path, "r") as f:
                            collected_data = json.load(f)
                        if collected_data.get("mayil_collected_data", {}).get("status", "") == "completed":
                            current_cost += sum(collected_data["ai_cost"].values())
                            continue
                    except:
                        pass
                filtered_task_batch.append(task)
            tasks_data = await asyncio.gather(*(process_task(task, testbed, ai_obj=ai_obj, db_obj=db_obj) for task in filtered_task_batch))
            for j, task_data in enumerate(tasks_data):
                task_status, collected_data = task_data
                logger.info(f"Task {j} status: {task_status}")
                if task_status != TaskReturnState.SUCCESS:
                    logger.error(f"Task {j} failed")
                else:
                    current_cost += sum(collected_data["ai_cost"].values())
                    with open(COLLECTION_DIR / f"{collected_data['id']}.json", "w") as f:
                        json.dump(collected_data, f, indent=4)
        # for task in enumerate(task_instances):
        #     if current_cost > 200:
        #         logger.error(f"Cost exceeded 200")
        #         break
        #     output_path = COLLECTION_DIR / f"{task['instance_id']}.json"
        #     if output_path.exists():
        #         try:
        #             with open(output_path, "r") as f:
        #                 collected_data = json.load(f)
        #             if collected_data.get("mayil_collected_data", {}).get("status", "") == "completed":
        #                 continue
        #         except:
        #             pass
            # if i % total_processes == process_index:
            #     # my_tasks.append((task, testbed))
            #     task_status, collected_data = await process_task(task, testbed, ai_obj=ai_obj, db_obj=db_obj)
            #     with open(COLLECTION_DIR / f"{collected_data['id']}.json", "w") as f:
            #         json.dump(collected_data, f, indent=4)
            #     current_cost += sum(collected_data["ai_cost"].values())
            #     logger.info(f"Process {process_index} | Task {i} status: {task_status}")
                

    # tasks_data = await asyncio.gather(*(process_task(task, testbed, ai_obj=ai_obj, db_obj=db_obj) for task, testbed in my_tasks))
    # for j, task_data in enumerate(tasks_data):
    #     task_status, collected_data = task_data
    #     with open(COLLECTION_DIR / f"{collected_data['id']}.json", "w") as f:
    #         json.dump(collected_data, f, indent=4)
    #     logger.info(f"Process {process_index} | Task {j} status: {task_status}")


if __name__ == "__main__":
    RedisClientSingleton()
    linear_token = os.getenv("LINEAR_TOKEN")
    if not linear_token:
        logger.error("LINEAR_TOKEN not found")

    LinearBot(
        key=linear_token
    )
    # logger.remove()

    # if len(sys.argv) == 3:
    #     total_processes = int(sys.argv[1])
    #     process_index = int(sys.argv[2])
    #     logger.add(f"./logs/{MAYIL_VERSION}_{process_index}.log", level="ERROR", colorize=False, backtrace=True, diagnose=True)
    # else:
    #     total_processes = 1
    #     process_index = 0
    #     logger.add(f"./logs/{MAYIL_VERSION}_{process_index}.log", level="ERROR", colorize=False, backtrace=True, diagnose=True)
    #     logger.info(f"Usage: {sys.argv[0]} <total_processes> <process_index>")
    #     logger.warning("Running in debug mode")
    sample_set = {
     "testbed/aftersell":[
        # "BEAM-2996",
        # https://github.com/ROKT/aftersell/pull/1083
        # "BEAM-3120",
        # https://github.com/ROKT/aftersell/pull/1129
        "BEAM-3220",
        "BEAM-3449",
        "BEAM-3375",
        "BEAM-3118",
        "BEAM-3125",
        "BEAM-3157",
        "BEAM-3254",
        "BEAM-3123",
        "BEAM-3124",
        "BEAM-3113",
        "BEAM-3064",
        "ENG-464",
        "OPS-4125"
    ],
    "testbed/UpCart-2.0":[
        "BEAM-2284",
        "BEAM-2750",
        "BEAM-2762",
    ]
    }
    
    distributed_tasks = []

    for repo_name, issue_list in sample_set.items():
        for issue_id in issue_list:
            distributed_tasks.append({
                "testbed": f"./{repo_name}",
                "task_instances": [
                    {
                        "repo_name": repo_name,
                        "instance_id": issue_id,
                    }
                ]
            })

    if not COLLECTION_DIR.exists():
        COLLECTION_DIR.mkdir(parents=True)
    
    asyncio.run(main(distributed_tasks))