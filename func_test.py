import subprocess

def check_compilation(so_name, file_name):
    """
    This function attempts to compile a file using the NVIDIA CUDA compiler (nvcc) and returns
    the result of the compilation process.

    Parameters:
    so_name (str): The name of the output shared object file (.so).
    file_name (str): The name of the file to be compiled.

    Returns:
    tuple: A tuple containing a boolean indicating success or failure, and the output or error message.
    """
    try:
        output = subprocess.run(
            ["nvcc", "-Xcompiler", "-fPIC", "-shared", "-o", so_name, file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def check_runtime(file_name, test_file):
    """
    Executes a runtime test on a specific file using a designated test script and returns the result of the execution.
    
    Parameters:
    file_name (str): The name of the file to be tested.
    test_file (str): The name of the Python script file containing the test logic.
    
    Returns:
    tuple: A tuple containing a boolean and the output (or error message).
    """
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,           
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output

def run_tests(file_name, test_file):
    """
    Executes a runtime test on a specific file using a designated test script and returns the result of the execution.
    
    Parameters:
    file_name (str): The name of the file to be tested.
    test_file (str): The name of the Python script file containing the test logic.
    
    Returns:
    tuple: A tuple containing a boolean and the output (or error message).
    """
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,           
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output



def perf_test():
    """Read the kernel launch time from the saved runtime file."""
    #TODO:
    return None
