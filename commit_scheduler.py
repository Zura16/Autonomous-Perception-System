import os
import sys
import json
import subprocess
import argparse

STATE_FILE = "commit_state.json"

# Define the groups of files to push daily (~15% of files per day)
FILE_GROUPS = [
    ["requirements.txt", ".gitignore"],
    ["config.py", "generate_synthetic_video.py"],
    ["src/__init__.py", "src/utils.py", "src/video_stream.py"],
    ["src/detector.py", "src/tracker.py"],
    ["src/lane_detector.py", "src/depth_estimator.py"],
    ["src/decision_logic.py", "main.py"],
    ["tests/__init__.py", "tests/test_pipeline.py", "README.md", "commit_scheduler.py", "commit_state.json"],
    ["requirements.txt"],
    ["src/web_stream_handler.py"],
    ["main_web.py"],
    ["web/index.html"],
    ["web/styles.css"],
    ["web/app.js"]
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"current_day": 0, "remote_url": "https://github.com/Zura16/Autonomous-Perception-System.git"}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command '{cmd}':\n{result.stderr.strip()}")
        return False, result.stderr.strip()
    return True, result.stdout.strip()

def initialize_git(remote_url=None):
    print("Checking git repository initialization...")
    if not os.path.exists(".git"):
        success, _ = run_command("git init")
        if not success:
            return False
        run_command("git branch -M main")
        print("Initialized new git repository with branch 'main'.")
    else:
        print("Git repository already exists.")

    if remote_url:
        _, remotes = run_command("git remote")
        if "origin" in remotes:
            run_command("git remote remove origin")
        success, _ = run_command(f"git remote add origin {remote_url}")
        if success:
            print(f"Set remote origin to: {remote_url}")
            return True
        return False
    return True

def push_daily_increment():
    state = load_state()
    day = state["current_day"]
    remote_url = state["remote_url"]

    # 1. Check if git is initialized
    initialize_git(remote_url if remote_url else None)

    # 2. Check if we have completed all days
    if day >= len(FILE_GROUPS):
        print(f"All {len(FILE_GROUPS)} days of file uploads have been processed! Git history is fully aligned.")
        return

    files_to_add = FILE_GROUPS[day]
    print(f"\n--- RUNNING RELEASE: DAY {day + 1} of {len(FILE_GROUPS)} ---")
    print(f"Files to commit: {', '.join(files_to_add)}")

    # Verify if files exist locally before committing
    missing_files = [f for f in files_to_add if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: The following files are missing and will be skipped: {', '.join(missing_files)}")
        files_to_add = [f for f in files_to_add if os.path.exists(f)]

    if not files_to_add:
        print("No files available to commit for today.")
        return

    # 3. Stage, commit, and push each file separately
    for file in files_to_add:
        print(f"Staging and committing: {file}...")
        run_command(f"git add '{file}'")
        commit_msg = f"[Day {day + 1}/7 Release] Adding {os.path.basename(file)}"
        success, stdout = run_command(f'git commit -m "{commit_msg}"')
        if not success:
            if "nothing to commit" in stdout or "clean" in stdout:
                print(f"File {file} already committed or no changes.")
            else:
                print(f"Commit failed for {file}.")
                return

        if remote_url:
            print(f"Pushing {file} to remote...")
            push_success, push_out = run_command("git push origin main")
            if not push_success:
                print(f"Push failed for {file}!")
                print(f"Detailed error: {push_out}")
                return

    # 4. Update state locally
    state["current_day"] = day + 1
    save_state(state)

    # 5. Stage, commit, and push the updated state file separately
    print("Staging and committing commit_state.json...")
    run_command(f"git add '{STATE_FILE}'")
    state_commit_msg = f"[Day {day + 1}/7 Release] Updating release state to Day {day + 2}"
    success, stdout = run_command(f'git commit -m "{state_commit_msg}"')
    
    if success and remote_url:
        print("Pushing commit_state.json update to remote...")
        push_success, push_out = run_command("git push origin main")
        if not push_success:
            print("Failed to push commit_state.json update!")
            print(f"Detailed error: {push_out}")
            return
            
    print(f"Day {day + 1} release complete. Next run will be Day {day + 2}.")

def main():
    parser = argparse.ArgumentParser(description="Automate 15% daily incremental pushes to GitHub.")
    parser.add_argument("--set-remote", type=str, help="Configure or update the remote GitHub repository URL")
    parser.add_argument("--run", action="store_true", help="Execute the scheduled commit and push for today")
    parser.add_argument("--status", action="store_true", help="Display the current scheduling release status")
    
    args = parser.parse_args()
    state = load_state()

    if args.set_remote:
        url = args.set_remote.strip()
        state["remote_url"] = url
        save_state(state)
        initialize_git(url)
        print(f"Remote URL configured: {url}")
        
    elif args.status:
        day = state["current_day"]
        remote = state["remote_url"] or "Not configured"
        print("=" * 45)
        print("      COMMIT & PUSH RELEASE SCHEDULER")
        print("=" * 45)
        print(f"Current Day: {day} of {len(FILE_GROUPS)}")
        print(f"Remote URL : {remote}")
        if day < len(FILE_GROUPS):
            print(f"Next Files : {', '.join(FILE_GROUPS[day])}")
            percentage = (day / len(FILE_GROUPS)) * 100
            print(f"Completion : {percentage:.1f}%")
        else:
            print("Completion : 100.0% (All days complete)")
        print("=" * 45)
        
    elif args.run:
        push_daily_increment()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
