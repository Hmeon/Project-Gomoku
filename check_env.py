import sys
import importlib.util

print(f"Python Version: {sys.version}")

def check_package(name):
    if importlib.util.find_spec(name):
        try:
            lib = importlib.import_module(name)
            print(f"SUCCESS: '{name}' is installed. Version: {getattr(lib, '__version__', 'unknown')}")
            return lib
        except ImportError as e:
            print(f"ERROR: '{name}' is installed but could not be imported. {e}")
    else:
        print(f"MISSING: '{name}' is NOT installed.")
    return None

torch_lib = check_package("torch")
dml_lib = check_package("torch_directml")

if torch_lib and dml_lib:
    try:
        device = dml_lib.device()
        print(f"DirectML Device Available: {device}")
        
        # Simple tensor test
        t = torch_lib.tensor([1, 2, 3]).to(device)
        print(f"Tensor on DirectML: {t}")
        print("DirectML Test PASSED!")
    except Exception as e:
        print(f"DirectML Test FAILED: {e}")
else:
    print("Skipping DirectML test due to missing packages.")
