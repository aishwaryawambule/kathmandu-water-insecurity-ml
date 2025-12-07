import subprocess
import os
import sys

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        # Run script from the 'src' directory so relative paths work
        result = subprocess.run(
            [sys.executable, script_name], 
            cwd="src", 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

def main():
    print("Starting ML Pipeline...")
    
    scripts = [
        "data_generation.py",
        "aquifer_clustering.py",
        "merge_data.py",
        "predict.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
