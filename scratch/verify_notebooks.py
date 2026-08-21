import os
import json
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def verify_json_format():
    notebook_files = glob.glob("04-deep-learning/foundations/**/*.ipynb", recursive=True)
    if not notebook_files:
        raise ValueError("No notebooks found under 04-deep-learning/foundations!")
        
    print(f"Found {len(notebook_files)} notebooks to verify.")
    
    for filepath in notebook_files:
        # 1. Check valid JSON syntax
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                nb_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error in {filepath}: {e}")
            return False
            
        # 2. Check basic nbformat keys
        required_keys = ["cells", "metadata", "nbformat", "nbformat_minor"]
        for key in required_keys:
            if key not in nb_dict:
                print(f"❌ Missing key '{key}' in notebook {filepath}")
                return False
                
        print(f"✓ JSON format ok: {filepath}")
    return True

def run_solutions():
    solution_files = sorted(glob.glob("04-deep-learning/foundations/**/*_solutions.ipynb", recursive=True))
    print(f"\nFound {len(solution_files)} solution notebooks to execute.")
    
    for filepath in solution_files:
        print(f"Executing {filepath} ...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
                
            ep = ExecutePreprocessor(timeout=180, kernel_name='python3')
            # Run the notebook in its own directory so local imports / relative paths work
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(filepath)}})
            print(f"✓ Execution success: {filepath}")
        except Exception as e:
            print(f"❌ Execution failure in {filepath}: {e}")
            return False
            
    return True

if __name__ == "__main__":
    print("=== STEP 1: Verifying Notebook JSON Formatting ===")
    json_ok = verify_json_format()
    if not json_ok:
        exit(1)
        
    print("\n=== STEP 2: Running Solutions Notebooks Execution ===")
    exec_ok = run_solutions()
    if not exec_ok:
        exit(1)
        
    print("\n🎉 All notebook validations completed successfully!")
