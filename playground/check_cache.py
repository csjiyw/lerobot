import diskcache

CACHE_DIR = "/root/_robot2_ckpt"

def main():
    # Initialize the cache
    cache = diskcache.Cache(CACHE_DIR)
    
    # 1. Check the last successful task ID
    cached_id = cache.get("cached_task_id")
    print(f"=== Resume Point ===")
    print(f"Last cached task ID: {cached_id}")
    print("\n" + "="*50 + "\n")

    # 2. Check for conversion failures
    print(f"=== Conversion Failures ===")
    failures = []
    for key in cache.iterkeys():
        if key.startswith("convert_failure_"):
            failures.append(key)
    
    if not failures:
        print("No failures found in cache.")
    else:
        print(f"Found {len(failures)} failures:\n")
        # Sort keys to show them in a consistent order
        for key in sorted(failures):
            task_id = key.replace("convert_failure_", "")
            data = cache.get(key)
            
            print(f"Task ID: {task_id}")
            print(f"  Device:      {data.get('device')}")
            print(f"  Robot:       {data.get('robot')}")
            print(f"  Return Code: {data.get('returncode')}")
            print(f"  Command:     {' '.join(data.get('cmd', []))}")
            print(f"  Traceback:\n{data.get('traceback')}")
            print(f"  stderr:\n{data.get('stderr')}")
            print("-" * 40)

if __name__ == "__main__":
    main()
