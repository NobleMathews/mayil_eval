import json
import os
import re
import subprocess
import tempfile
import time
import webbrowser
from datetime import datetime
from typing import Optional

import markdown2
import requests
import typer
from dotenv import load_dotenv
from git import Repo
from github import Github, GithubException
from loguru import logger
from mayil import MayilIssue
from pygments.formatters.html import HtmlFormatter

app = typer.Typer(
    help="This CLI tool helps create triggers to simulate GitHub issues handling"
)
load_dotenv(".env")
load_dotenv(".env.local")

GITHUB_TOKEN = os.getenv("TESTING_GITHUB_TOKEN")

if GITHUB_TOKEN is None:
    raise typer.Exit("Please set the TESTING_GITHUB_TOKEN environment variable.")

gh = Github(GITHUB_TOKEN)

LINK_TO_MAYIL = os.getenv("LINK_TO_MAYIL", "http://localhost:8080")

############### REPO HELPERS ################


def clone_repository(source_repo_url, local_dir):
    # Cloning the source repository with depth 1
    Repo.clone_from(source_repo_url, local_dir)


def create_or_get_repository(gh, repo_name):
    try:
        return gh.get_repo(repo_name)
    except Exception:
        # Repository does not exist, so create it
        org, repo_name = repo_name.split("/")
        try:
            organization = gh.get_organization(org)
            return organization.create_repo(repo_name)
        except Exception:
            user = gh.get_user()
            return user.create_repo(repo_name)


def push_to_repository(local_dir, test_repo_url):
    # Setting up the new repository as a remote and force pushing to it
    subprocess.run(
        ["git", "-C", local_dir, "remote", "set-url", "origin", test_repo_url],
        check=True,
    )
    subprocess.run(["git", "-C", local_dir, "branch", "-M", "mayil"], check=True)
    subprocess.run(
        ["git", "-C", local_dir, "push", "-u", "--force", "origin", "mayil"], check=True
    )


############### ISSUE HELPERS ################


def create_github_issue(repo, title, body):
    try:
        issue = repo.create_issue(
            title=_remove_links_and_mentions_from_body(title),
            body=_remove_links_and_mentions_from_body(body),
        )
        logger.info("Successfully created Issue:", issue.number)
    except GithubException as e:
        logger.opt(exception=e).error("Could not create Issue")


def close_issue(repo, issue_id):
    try:
        issue = repo.get_issue(number=issue_id)
        issue.edit(state="closed")
        logger.info(f"Issue #{issue_id} closed.")
    except GithubException as e:
        logger.opt(exception=e).error(f"Error closing issue #{issue_id}: {e}")


def close_all_issues(repo):
    open_issues = repo.get_issues(state="open")
    for issue in open_issues:
        close_issue(repo, issue.number)


def get_issues_from_date(repo, date):
    """
    date should be YYYY-MM-DD. For example, 2020-01-01
    """
    # remove pull requests
    issues = repo.get_issues(state="closed", since=datetime.fromisoformat(date))
    for issue in issues:
        if not issue.pull_request:
            yield issue


def fetch_issue_details(owner, repo, issue_number):
    try:
        repository = gh.get_repo(f"{owner}/{repo}")
        issue = repository.get_issue(number=int(issue_number))
        return issue.title, issue.body
    except GithubException as e:
        logger.opt(exception=e).error(f"Error fetching issue details: {e}")
        return None, None


############### COMMIT HELPERS ################
def create_dummy_change_and_push_to_github(
    source_repo, file_to_change, data_to_append, commit_message
):
    """ "
    Used for integration tests

    Appends data_to_append to file_to_change and commits the change to the repository with commit_message as the commit message and pushes the change to the repository.
    """
    # Clone the repo to a temporary directory
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{source_repo}"
    repo = Repo.clone_from(
        repo_url.replace("github.com/", f"x-access-token:{GITHUB_TOKEN}@github.com/"),
        temp_dir,
    )

    repo.git.config("user.email", "support@mayil.ai")
    repo.git.config("user.name", "Mayil AI")

    # Modify a file
    file_path = os.path.join(repo.working_tree_dir, file_to_change)
    with open(file_path, "a") as file:
        file.write(data_to_append)

    # Commit the change
    repo.git.add(file_to_change)
    repo.git.commit("-m", commit_message)

    # Push the change
    repo.git.push("origin", "HEAD")

    # Optionally, get the SHA of the latest commit
    commit_sha = repo.head.commit.hexsha

    return commit_sha


def delete_most_recent_commit_from_github(source_repo):
    """
    Used for integration tests

    Deletes the most recent commit from the repository.
    """

    # cone the repo to a temp directory
    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)
    clone_repository(f"https://github.com/{source_repo}", temp_dir.name)

    # delete the most recent commit

    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
    subprocess.run(["git", "push", "origin", "HEAD", "--force"], check=True)


def checkout_to_commit_just_before_date(local_dir, date):
    command_find_main_branch = (
        "git -C {} remote show origin | grep 'HEAD branch' | cut -d' ' -f5".format(
            local_dir
        )
    )
    main_branch = subprocess.check_output(
        command_find_main_branch, shell=True, text=True
    ).strip()

    # Finding the last commit before the specified date on the main branch
    command_find_commit = "git -C {} rev-list -1 --before='{}' {}".format(
        local_dir, date, main_branch
    )
    result = subprocess.run(
        command_find_commit, shell=True, text=True, capture_output=True
    )
    commit_to_checkout = result.stdout.strip()

    command_checkout = "git -C {} checkout -b mayil {}".format(
        local_dir, commit_to_checkout
    )
    result = subprocess.run(
        command_checkout, shell=True, text=True, capture_output=True
    )
    print("Checkout stdout:", result.stdout.strip())


############### HTML LOGIC ################


def render_markdown_to_html(issue_md, content, output_filename="test.html"):
    """
    Renders the given markdown text as HTML with GitHub-like styling.
    The issue_md appears as a GitHub issue and content appears like a comment.

    :param issue_md: Markdown string for the issue
    :param content: Markdown string for the comment
    :param output_filename: The name of the output HTML file
    """

    # Generate Pygments CSS for light and dark themes
    formatter_light = HtmlFormatter(style="default")
    pygments_css_light = formatter_light.get_style_defs(".codehilite")

    formatter_dark = HtmlFormatter(style="monokai")
    pygments_css_dark = formatter_dark.get_style_defs(".codehilite.dark-mode")

    issue_html = markdown2.markdown(
        issue_md, extras=["fenced-code-blocks", "code-friendly", "toc"]
    )
    comment_html = markdown2.markdown(
        content, extras=["fenced-code-blocks", "code-friendly", "toc"]
    )

    # HTML template with GitHub-like CSS, dark mode toggle, and styled button
    html_template = f"""<!DOCTYPE html>
    <html>
    <head>
    <link id="styleSheet" rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.4.0/github-markdown-light.min.css">
    <style>
        .codehilite{{
            background: None !important;
        }}
        {pygments_css_light}
        {pygments_css_dark}
        body {{
            position: relative;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            transition: background-color 0.3s, color 0.3s;
            width: 90%;
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }}
        .comment {{
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background-color: #f6f8fa;
        }}
        .dark-mode .comment {{
            border-color: #30363d;
            background-color: #0d1117;
        }}
        @media (max-width: 767px) {{
            .markdown-body {{
                padding: 15px;
            }}
        }}
        .theme-switch {{
            position: absolute;
            top: 16px;
            right: 16px;
            display: inline-block;
            width: 50px;
            height: 25px;
            background: #f6f8fa;
            border-radius: 25px;
            border: 1px solid #ddd;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        .theme-switch:before {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 21px;
            height: 21px;
            background-color: #FDB813;
            border-radius: 50%;
            transition: 0.3s;
        }}
        .theme-switch.dark-mode:before {{
            transform: translateX(25px);
            background-color: #333;
        }}
        body.dark-mode {{
            background-color: #0d1117;
        }}
        blockquote {{
        padding: 0.5em 10px;
        margin: 1em 0;
        border-left: 0.25em solid #dfe2e5;
        background-color: #f6f8fa;
        color: #24292e;
        }}
        .dark-mode blockquote {{
            background-color: #0d1117;
            border-left-color: #30363d;
            color: #c9d1d9;
        }}
        .copy-code {{
            position: relative;
        }}
        .copy-button {{
            position: absolute;
            top: 0.5em;
            right: 0.5em;
            z-index: 10;
            padding: 0.2em 0.5em;
            font-size: 1.0em;
            color: #333;
            background-color: #f9f9f9;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            opacity: 0.7; /* Make the button slightly transparent */
        }}
        .copy-button:hover {{
            opacity: 1; /* Full visibility on hover */
        }}
        .dark-mode .copy-button {{
            background-color: #30363d;
            color: #c9d1d9;
        }}
    </style>
    </head>
    <body>
    <div class="theme-switch" onclick="toggleDarkMode()"></div>
    <article class="markdown-body">
            {issue_html}
    </article>
    <article class="markdown-body comment">
        {comment_html}
    </article>
    <script>
        function toggleDarkMode() {{
            var stylesheet = document.getElementById('styleSheet');
            var themeSwitch = document.querySelector('.theme-switch');
            if (stylesheet.href.includes('github-markdown-light.min.css')) {{
                stylesheet.href = 'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.4.0/github-markdown-dark.min.css';
                themeSwitch.classList.add('dark-mode');
            }} else {{
                stylesheet.href = 'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.4.0/github-markdown-light.min.css';
                themeSwitch.classList.remove('dark-mode');
            }}
            var body = document.body;
            body.classList.toggle('dark-mode');
            document.querySelectorAll('.codehilite').forEach(function(element) {{
                element.classList.toggle('dark-mode');
            }});
        }}
        // Function to copy code to clipboard
    function copyToClipboard(text) {{
        var dummy = document.createElement('textarea');
        document.body.appendChild(dummy);
        dummy.value = text;
        dummy.select();
        document.execCommand('copy');
        document.body.removeChild(dummy);
    }}

    // Add copy buttons to code blocks
    document.addEventListener('DOMContentLoaded', function() {{
        var codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(function(codeBlock) {{
            var copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = 'Copy';
            copyButton.addEventListener('click', function() {{
                copyToClipboard(codeBlock.textContent);
            }});
            var pre = codeBlock.parentNode;
            pre.style.position = 'relative';
            pre.insertBefore(copyButton, pre.firstChild);
        }});
    }});
    </script>
    </body>
    </html>"""

    # Write to an HTML file

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        # Write the HTML template to the file
        f.write(html_template.encode("utf-8"))

        # Open in the default browser
        webbrowser.open("file://" + f.name)


############### MAYIL HELPERS ################
def call_mayil_inspector(
    event, action, params=None, mayil_url=LINK_TO_MAYIL + "/inspector/"
):
    # Define the data to send
    data = {
        "event": event,
        "action": action,
        "params": params,
    }
    # Send the POST request
    response = requests.post(
        mayil_url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    )
    status, response = response.json()

    if status != 200:
        logger.error(f"Request Failed (Status {status}): {response}")
        return False
    else:
        logger.info(f"Server Response {status}: {response}")
        return response


def call_mayil_test_endpoint(
    id,
    repo_name,
    title,
    body,
    mayil_url=LINK_TO_MAYIL + "/test/",
    requested_commit="",
):
    # Define the data to send
    data = {
        "id": id,
        "repo_name": repo_name,
        "repo_link": f"https://github.com/{repo_name}",
        "title": title,
        "body": body,
        "requested_commit": requested_commit,
    }

    print(data)

    # Send the POST request
    response = requests.post(
        mayil_url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    )
    status, response = response.json()

    if status != 200:
        logger.error(f"Request Failed (Status {status}): {response}")
    else:
        logger.info(f"Server Response {status}: {response}")
    return int(response)


def get_mayil_response(
    task_id: str,
    issue_md: str = "",
    retry_timeout: int = 30,
    max_attempts=10,
    render_html=False,
    url=LINK_TO_MAYIL + "/get-results/",
):
    logger.info(f"Checking task status at {url}")

    if url[-1] != "/":
        url += "/"
    url = f"{url}{task_id}"

    status = "Does not exist"
    attempt = 0

    while status == "Does not exist" or status == "processing":
        attempt += 1
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Request Failed (Status {response.status_code})")
        else:
            server_response = response.json()
            status = server_response["status"]
            if status == "Does not exist" or status == "processing":
                logger.info(
                    f"Task is still running. Retrying in {retry_timeout} seconds"
                )
                if attempt >= max_attempts:
                    raise Exception("Max attempts reached")
                time.sleep(retry_timeout)
            elif status == "failed":
                raise Exception(f"Task failed: {server_response['result']}")
            elif status == "completed":
                logger.info("Task completed")
                content = server_response["result"]
                if render_html:
                    render_markdown_to_html(issue_md, content)
                else:
                    return content

            else:
                logger.error(f"Unknown status: {status}")
                raise Exception(f"Unknown status: {status}")


############### HELPER'S HELPERS ################
def _remove_links_and_mentions_from_body(body):
    # remove links
    body = re.sub(r"http\S+", "", body)
    body = re.sub(r"@[^\s]+", "", body)
    # replace all "#" with "hashtag"
    body = re.sub(r"#", "hashtag", body)
    return body


def _process_url(file_or_url: str) -> MayilIssue:
    """
    Process the file_or_url argument to extract issue details.

    :param file_or_url: A file path or a URL to an existing GitHub issue.
    :return: MayilIssue containing the extracted issue information.
    """
    match = re.match(r"https?://github\.com/(.+)/(.+)/issues/(\d+)", file_or_url)
    if match:
        owner, repo, issue_number_str = match.groups()
        issue_title, issue_body = fetch_issue_details(
            owner, repo, int(issue_number_str)
        )
        repo_name = f"{owner}/{repo}"
        return issue_number_str, repo_name, issue_title, issue_body
    else:
        raise typer.Exit("Not Implemented: Please provide a valid GitHub issue URL.")


def create_issue_from_date(test_repo, source_repo, date, run_locally=False):
    for issue in get_issues_from_date(source_repo, date):
        logger.info(f"Creating issue for {issue.title} (#{issue.number})")
        title = f"{issue.title} (#{issue.number})"
        body = f"{issue.body}\n)"
        if run_locally:
            return call_mayil_test_endpoint(
                int(issue.number),
                source_repo.full_name,
                title,
                body,
                LINK_TO_MAYIL + "/test/",
            )
        else:
            create_github_issue(test_repo, title, body)


def batch_process(
    test_repo: str, source_repo: str, date: str, run_locally=False, upload_repo=False
):
    test_repository = None
    if not run_locally:
        test_repository = create_or_get_repository(gh, test_repo)
        source_repo_url = f"https://github.com/{source_repo}.git"
        test_repo_url = test_repository.clone_url
        if upload_repo:
            with tempfile.TemporaryDirectory() as local_dir:
                clone_repository(source_repo_url, local_dir)
                checkout_to_commit_just_before_date(local_dir, date)
                push_to_repository(local_dir, test_repo_url)

    source_repository = gh.get_repo(source_repo)
    create_issue_from_date(test_repository, source_repository, date, run_locally)


############### CLI ################


@app.command(help="Close one or all issues in a GitHub repository.")
def close(
    github_repository: str = typer.Argument(
        ..., help="The name of the GitHub repository."
    ),
    issue_id: Optional[int] = typer.Option(
        None,
        help="The ID of a specific issue to close. If not provided, all issues will be closed.",
    ),
):
    test_repo = gh.get_repo(github_repository)
    if issue_id is not None:
        close_issue(test_repo, issue_id)
    else:
        close_all_issues(test_repo)


@app.command(
    help="Create issues in a test repository based on issues from a source repository starting from a specific date."
)
def batch(
    test_repo: str = typer.Argument(
        ..., help="The name of the test GitHub repository."
    ),
    source_repo: str = typer.Argument(
        ..., help="The name of the source GitHub repository."
    ),
    date: str = typer.Argument(
        ..., help="The start date for creating issues, in YYYY-MM-DD format."
    ),
    upload_repo: bool = typer.Option(
        False, help="Upload the test repository to GitHub."
    ),
):
    batch_process(
        test_repo=test_repo,
        source_repo=source_repo,
        date=date,
        run_locally=False,
        upload_repo=upload_repo,
    )


@app.command(
    help="Call local endpoint based on issues from a source repository starting from a specific date."
)
def local_batch(
    source_repo: str = typer.Argument(
        ..., help="The name of the source GitHub repository."
    ),
    date: str = typer.Argument(
        ..., help="The start date for creating issues, in YYYY-MM-DD format."
    ),
):
    batch_process(
        test_repo="",
        source_repo=source_repo,
        date=date,
        run_locally=True,
    )


@app.command(
    help="Create a new issue in a GitHub repository, either from a file or URL."
)
def create(
    github_repository: str = typer.Argument(
        ..., help="The name of the GitHub repository where the issue will be created."
    ),
    file_or_url: str = typer.Argument(
        ..., help="A file path or a URL to an existing GitHub issue."
    ),
):
    test_repo = gh.get_repo(github_repository)
    issue_info = _process_url(file_or_url)

    if issue_info.title and issue_info.body:
        create_github_issue(test_repo, issue_info.title, issue_info.body)
    else:
        raise typer.Exit(
            "Invalid issue command. Please provide a valid file path or GitHub issue URL."
        )


@app.command(
    help="Call the Mayil test endpoint with details  either from a file or from a GitHub issue."
)
def local(
    file_or_url: str = typer.Argument(
        ..., help="A file path or a URL to an existing GitHub issue."
    ),
    view_response: bool = typer.Option(
        True, help="Render the response as Markdown in the default browser."
    ),
    view_as_html: bool = typer.Option(
        True, help="Render the response as HTML in the default browser."
    ),
):
    id, repo_name, issue_title, issue_body = _process_url(file_or_url)

    if issue_title and issue_body:
        if repo_name:
            task_id = call_mayil_test_endpoint(
                id, repo_name, issue_title, issue_body, LINK_TO_MAYIL + "/test/"
            )
        else:
            raise typer.Exit(
                "Not Implemented: Please provide a valid GitHub issue URL."
            )
    else:
        raise typer.Exit(
            "Invalid issue command. Please provide a valid file path or GitHub issue URL."
        )

    if view_response:
        logger.success(
            get_mayil_response(task_id, issue_body, render_html=view_as_html)
        )

    return


if __name__ == "__main__":
    app()
