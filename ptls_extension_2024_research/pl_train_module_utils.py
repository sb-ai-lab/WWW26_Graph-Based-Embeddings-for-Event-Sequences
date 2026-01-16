import subprocess

def get_git_commit_hash():
    command = "git show -s --format=%H"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

