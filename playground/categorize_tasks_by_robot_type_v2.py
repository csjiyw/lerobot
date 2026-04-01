# -*- coding: utf-8 -*-
"""
Categorize task folders by robot_type from /mnt/fastdisk/v21 to /mnt/fastdisk/v21_category

Workflow:
1. Find task ID folders in SOURCE_ROOT
2. Read robot_type from meta/info.json
3. Create robot_type folder in DEST_ROOT if not exists
4. Copy task folder to DEST_ROOT/robot_type/ folder
"""

import os
import json
import shutil
import subprocess
from tqdm import tqdm
import traceback

# ====== CONFIG ======
SOURCE_ROOT = "/mnt/fastdisk/v21"
DEST_ROOT = "/mnt/fastdisk/v21_category"
# =====================


def get_robot_type(task_path: str) -> str:
    """
    Read robot_type from meta/info.json
    
    Args:
        task_path: Path to task folder (e.g., /mnt/fastdisk/v21/fe2e604a1f5746d3a8c6db589367df70)
    
    Returns:
        robot_type string (e.g., "QinLongROS1") or "Unknown" if not found
    """
    info_path = os.path.join(task_path, "meta", "info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        robot_type = data.get("robot_type")
        if robot_type:
            return str(robot_type)
        else:
            print(f"  [WARN] No robot_type in {info_path}")
            return "Unknown"
    except FileNotFoundError:
        print(f"  [WARN] meta/info.json not found in {task_path}")
        return "Unknown"
    except json.JSONDecodeError as e:
        print(f"  [WARN] Invalid JSON in {info_path}: {e}")
        return "Unknown"
    except Exception as e:
        print(f"  [WARN] Error reading {info_path}: {e}")
        return "Unknown"


def is_task_folder(folder_name: str) -> bool:
    """
    Check if folder looks like a task folder (hex string)
    
    Args:
        folder_name: Name of folder to check
    
    Returns:
        True if looks like a task folder
    """
    # Task folders are hex strings like "fe2e604a1f5746d3a8c6db589367df70"
    # Robot folders are names like "QinLongROS1"
    return folder_name.isalnum() and len(folder_name) >= 10


def categorize_tasks():
    """
    Main categorization workflow
    """
    print(f"Scanning {SOURCE_ROOT} for task folders...")
    print(f"Destination: {DEST_ROOT}")
    
    # Create destination root if not exists
    os.makedirs(DEST_ROOT, exist_ok=True)
    
    # 1. Find all task folders (skip robot type folders)
    all_items = os.listdir(SOURCE_ROOT)
    task_folders = [d for d in all_items 
                    if is_task_folder(d) 
                    and os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Found {len(task_folders)} task folders")
    
    if not task_folders:
        print("No task folders found. Exiting.")
        return
    
    # 2. Process each task
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for task_id in tqdm(task_folders, desc="Categorizing tasks"):
        task_path = os.path.join(SOURCE_ROOT, task_id)
        
        # 3. Get robot_type
        robot_type = get_robot_type(task_path)
        
        # 4. Create robot folder in DEST_ROOT if needed
        robot_folder = os.path.join(DEST_ROOT, robot_type)
        os.makedirs(robot_folder, exist_ok=True)
        
        # 5. Copy task to robot folder
        dest_path = os.path.join(robot_folder, task_id)
        
        try:
            if os.path.exists(dest_path):
                print(f"  [SKIP] {task_id} already exists in {robot_type}/")
                skip_count += 1
                continue
            
            print(f"  [COPY] {task_id} -> {robot_type}/")
            # shutil.copytree(task_path, dest_path)
            # subprocess.run(["rsync", "-a", task_path + "/", dest_path], check=True)
            subprocess.run(["mv", task_path, dest_path], check=True)
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to copy {task_id}: {e}")
            traceback.print_exc()
            error_count += 1
    
    # Summary
    print("\n" + "="*50)
    print("Categorization Complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print("="*50)


def main():
    """Entry point"""
    # Check source directory exists
    if not os.path.isdir(SOURCE_ROOT):
        print(f"ERROR: Source directory not found: {SOURCE_ROOT}")
        return
    
    categorize_tasks()


if __name__ == "__main__":
    main()
