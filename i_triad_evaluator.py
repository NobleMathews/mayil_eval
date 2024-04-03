from mayil.integrations.openai import OpenAI, DeployedModel
from mayil.integrations.redis import RedisClientSingleton
from textwrap import dedent
import re
import numpy as np
import asyncio
from pathlib import Path
import json
import sys
from tqdm.asyncio import tqdm

from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))

MAYIL_VERSION = "v1"
COLLECTION_DIR = Path(f"./data/RAG_triad_eval/{MAYIL_VERSION}")

ai_obj = OpenAI()

def re_0_10_rating(str_val: str) -> int:
    matches = re.compile(r"\b([0-9]|10)(?=\D*$|\s*\.)").fullmatch(str_val)
    if not matches:
        matches = re.search(r'([0-9]+)(?=\D*$)', str_val)
        if not matches:
            return -10
    return int(matches.group())

async def get_context_relevance(issue:str, ret:str):
    context = ret
    question = issue
    system_prompt = dedent(
        """
        You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

        A few additional scoring guidelines:

        - Long CONTEXTS should score equally well as short CONTEXTS.

        - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

        - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

        - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

        - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

        - Never elaborate.
        """
    )

    user_prompt = dedent(
        """
        QUESTION: {question}

        CONTEXT: {context}
            
        RELEVANCE: 
        """
    ).format(question=question, context=context)

    response,_,_ = await ai_obj.chat_completion(DeployedModel.GPT4, system_prompt, user_prompt)
    
    return re_0_10_rating(response) / 10

async def get_claims(issue:str, response:str):
    system_prompt = dedent(
        """
        You are an ungrounded claims extractor, 
        Reasonable assumptions / suggestions can be ignored as long as they are not blatantly false or contradicted by the available information.
        Extract such claims from the given RESPONSE which are not grounded in the CONTEXT.
        Respond with a list of such claims from the given CONTEXT.

        You will be provided with a CONTEXT and a RESPONSE.
        The CONTEXT describes a software issue or a problem statement.
        The RESPONSE is a generated response to the CONTEXT by a beginner who tends to make confident ungrounded claims.
        The RESPONSE may contain claims and statements that have no basis and are just assumption.
        If you find statements or claims that cannot be inferred from the CONTEXT, please list them.

        Each claim or statement should be a single sentence starting with a - (dash) and ending with a period. 
        They must all be concise but comprehensive enough to understand without looking at the CONTEXT or other claims.
        For example:
        CONTEXT: The user is unable to login to the application, it shows an error message "Invalid credentials". I am connected to the internet.
        RESPONSE: Assuming the server is using uvicorn, it maybe down. Please ensure that you are connected to the internet. Ensure that the password is correct.
        UNFOUNDED CLAIMS IN RESPONSE:
        - The server is using uvicorn.
        - User is not connected to the internet.

        With respect to modifications suggested to code claims can also include assumptions about the codebase, existing code, or the environment that are not explicitly or implicitly mentioned in the context.
        """
    )

    user_prompt = dedent(
        """
        CONTEXT: {context}

        RESPONSE: {response}
            
        UNFOUNDED CLAIMS IN RESPONSE: 
        """
    ).format(context=issue, response=response)

    response,_,_ = await ai_obj.chat_completion(DeployedModel.GPT4, system_prompt, user_prompt)

    ret = []
    for claim in response.splitlines():
        if not claim:
            continue
        if not claim.startswith("-"):
            continue
        ret.append(claim.strip())
    
    return ret

async def get_groundedness(hypotheses: list[str], premise:str):
    system_prompt= dedent(
        """
        You are a INFORMATION OVERLAP classifier; providing the overlap of information between the source and statement.
        Respond only as a number from 0 to 10 where 0 is no information overlap and 10 is all information is overlapping.
        Never elaborate.
        """
    )
    user_prompt= dedent(
        """
        SOURCE: {premise}
        
        Hypothesis: {hypothesis}
        
        Please answer with the template below for all statement sentences:

        Supporting Evidence: <Choose the exact parts of the source that can help validate the hypothesis, if nothing matches, say NOTHING FOUND>
        Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>
        """
    )

    groundedness_scores = {}
    reasons = {}
    for i, hypothesis in enumerate(hypotheses):
        reason,_,_ = await ai_obj.chat_completion(DeployedModel.GPT4, system_prompt, user_prompt.format(premise=premise, hypothesis=hypothesis))
        score_line = next((line for line in reason.split('\n') if "Score" in line), None)
        if score_line:
            groundedness_scores[f"statement_{i}"] = re_0_10_rating(score_line) / 10
            reasons[f"statement_{i}"] = reason
    return groundedness_scores, reasons


def grounded_statements_aggregator(
        source_statements_multi_output: list[dict]
) -> float:
    all_results = []

    statements_to_scores = {}

    if not isinstance(source_statements_multi_output, list):
        source_statements_multi_output = [source_statements_multi_output]

    for multi_output in source_statements_multi_output:
        for k in multi_output:
            if k not in statements_to_scores:
                statements_to_scores[k] = []
            statements_to_scores[k].append(multi_output[k])

    for k in statements_to_scores:
        all_results.append(np.max(statements_to_scores[k]))

    return np.mean(all_results)

async def get_qa_relevance(prompt:str, response:str):
    system_prompt= dedent(
        """
        You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

        A few additional scoring guidelines:

        - Long RESPONSES should score equally well as short RESPONSES.

        - Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the most RELEVANT.

        - RESPONSE must be relevant to the entire PROMPT to get a score of 10.

        - RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

        - RESPONSE that is RELEVANT to none of the PROMPT should get a score of 0.

        - RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

        - RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

        - RESPONSE that confidently FALSE should get a score of 0.

        - RESPONSE that is only seemingly RELEVANT should get a score of 0.

        - Never elaborate.
        """
    )

    user_prompt=dedent(
        """
        PROMPT: {prompt}

        RESPONSE: {response}

        Please answer using the entire template below.

        TEMPLATE: 
        Score: <The score 0-10 based on the given criteria>
        Criteria: <Provide the criteria for this evaluation>
        Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
        """
    ).format(prompt=prompt, response=response)

    response,_,_ = await ai_obj.chat_completion(DeployedModel.GPT4, system_prompt, user_prompt)

    if "Supporting Evidence" in response:
        score = -1
        supporting_evidence = None
        criteria = None
        for line in response.split('\n'):
            if "Score" in line:
                score = re_0_10_rating(line) / 10
            criteria_lines = []
            supporting_evidence_lines = []
            collecting_criteria = False
            collecting_evidence = False

            for line in response.split('\n'):
                if "Criteria:" in line:
                    criteria_lines.append(
                        line.split("Criteria:", 1)[1].strip()
                    )
                    collecting_criteria = True
                    collecting_evidence = False
                elif "Supporting Evidence:" in line:
                    supporting_evidence_lines.append(
                        line.split("Supporting Evidence:", 1)[1].strip()
                    )
                    collecting_evidence = True
                    collecting_criteria = False
                elif collecting_criteria:
                    if "Supporting Evidence:" not in line:
                        criteria_lines.append(line.strip())
                    else:
                        collecting_criteria = False
                elif collecting_evidence:
                    if "Criteria:" not in line:
                        supporting_evidence_lines.append(line.strip())
                    else:
                        collecting_evidence = False

            criteria = "\n".join(criteria_lines).strip()
            supporting_evidence = "\n".join(supporting_evidence_lines
                                            ).strip()
        reasons = {
            'reason':
                (
                    f"{'Criteria: ' + str(criteria)}\n"
                    f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                )
        }
        return score, reasons

    else:
        score = re_0_10_rating(response) / 10
        return score, {}
    
async def process_task(issue_details):
    generated_details_path = f"./data/v1/{issue_details['instance_id']}.json"
    if not Path(generated_details_path).exists():
        return
    output_path = COLLECTION_DIR / f"{issue_details['instance_id']}.json"
    if output_path.exists():
        with open(output_path, "r") as f:
            generated_details = json.load(f)
    else:
        with open(generated_details_path, "r") as f:
            generated_details = json.load(f)
    problem_statement = issue_details["problem_statement"]
    mayil_response = generated_details["mayil_response"]
    relevant_snippets = generated_details["mayil_collected_data"]["relevant_snippets"]

    if not mayil_response:
        return
    if len(generated_details["mayil_collected_data"]["relevant_snippets"])==0:
        return

    if not generated_details["mayil_collected_data"].get("context_relevance"): 
        snippet_relevance = []
        for c in relevant_snippets:
            ret = f"Filename: {c['filename']} | (Lines: {c['start_line']} to {c['end_line']})\nCode Snippet:\n{c['code']}"
            c["context_relevance"] = await get_context_relevance(problem_statement, ret)
            snippet_relevance.append(c["context_relevance"])
        context_relevance = np.mean(snippet_relevance)
        generated_details["mayil_collected_data"]["context_relevance"] = context_relevance

    if not generated_details["mayil_collected_data"].get("groundedness_scores"):
        hypotheses = await get_claims(problem_statement, mayil_response)
        rets = []
        for c in relevant_snippets:
            rets.append(f"Filename: {c['filename']} | (Lines: {c['start_line']} to {c['end_line']})\nCode Snippet:\n{c['code']}")
        premise = problem_statement + "---\n\n" + "---\n\n".join(rets)
        if hypotheses:
            groundedness_scores, reasons = await get_groundedness(hypotheses, premise)
            generated_details["mayil_collected_data"]["hypotheses"] = hypotheses
            generated_details["mayil_collected_data"]["groundedness_scores"] = groundedness_scores
            generated_details["mayil_collected_data"]["groundedness_reasons"] = reasons
            groundedness_score = grounded_statements_aggregator(groundedness_scores)
            generated_details["mayil_collected_data"]["groundedness_score"] = groundedness_score
        else:
            generated_details["mayil_collected_data"]["hypotheses"] = []
            generated_details["mayil_collected_data"]["groundedness_score"] = 1.0


    if not generated_details["mayil_collected_data"].get("qa_relevance"):
        qa_relevance, reasons = await get_qa_relevance(problem_statement, mayil_response)
        generated_details["mayil_collected_data"]["qa_relevance_reasons"] = reasons
        generated_details["mayil_collected_data"]["qa_relevance"] = qa_relevance

    with open(output_path, "w") as f:
        json.dump(generated_details, f, indent=4)

    

async def main(total_processes, process_index):

    RedisClientSingleton()

    with open("./data/test.json", "r") as f:
        ground_truth = json.load(f)
    
    task_instances = []
    for i, issue_details in enumerate(ground_truth):
        if i % total_processes == process_index:
            if total_processes == 1:
                task_instances.append(issue_details)
                break
            task_instances.append(issue_details)
    # my_batch = []
    for issue_details in tqdm(task_instances):
        await process_task(issue_details)
        # my_batch.append(issue_details)
        # if len(my_batch) == 50:
        #     await tqdm.gather(*(process_task(task) for task in my_batch))
        #     my_batch = []        

if __name__ == "__main__":
    if len(sys.argv) == 3:
        total_processes = int(sys.argv[1])
        process_index = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} <total_processes> <process_index>")
        total_processes = 1
        process_index = 0

    if not COLLECTION_DIR.exists():
        COLLECTION_DIR.mkdir(parents=True)
    
    asyncio.run(main(total_processes, process_index))

