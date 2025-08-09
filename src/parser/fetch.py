import os
import git

def fetch_repo(repo_url, branch_or_commit, local_path="repo_temp"):
    """
    Clones or fetches a git repository and checks out a specific branch, commit, or tag.

    :param repo_url: URL of the git repository
    :param branch_or_commit: Branch name, commit hash, or tag
    :param local_path: Local folder to clone into
    :return: Path to the local repo
    """
    if os.path.exists(local_path):
        print(f"Using existing repo at {local_path}")
        repo = git.Repo(local_path)
        repo.remotes.origin.fetch()
    else:
        print(f"Cloning {repo_url} into {local_path}...")
        repo = git.Repo.clone_from(repo_url, local_path)

    # Checkout the desired branch/commit/tag
    print(f"Checking out {branch_or_commit}...")
    repo.git.checkout(branch_or_commit)

    return local_path
