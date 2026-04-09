from typing import AnyStr
from .basetool import *
import builtins
import importlib
import signal

# Attention: This tool has no safety protection


class CodeInterpreter:
    def __init__(self, timeout=60):
        self.globals = {"__builtins__": builtins}
        self.locals = {}
        self.timeout = timeout

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Code execution timed out.")

    def execute_code(self, code):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set timeout handler
        signal.alarm(self.timeout)  # Start timer

        try:
            exec(code, self.globals, self.locals)
            return "Code executed successfully."
        except TimeoutError as e:
            return "Error: Code execution timed out."
        except NameError as e:
            missing_name = str(e).split("'")[1]  # Extract missing module name
            try:
                module = importlib.import_module(missing_name)  # Attempt to import
                self.globals[missing_name] = module  # Add to globals
                return f"Auto-imported module: {missing_name}. Try running again."
            except ModuleNotFoundError:
                return f"Error: {missing_name} is not a recognized module."
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            signal.alarm(0)  # Cancel the alarm

    def reset_session(self):
        self.globals = {"__builtins__": builtins} 
        self.locals = {}


class PythonCodeInterpreterArgs(BaseModel):
    code: str = Field(..., description="python codes")


class PythonCodeInterpreter(BaseTool):
    """Python Code Interpreter Tool"""
    name: str = "python_code_interpreter"  # Add type annotation
    description: str = "A tool to execute Python code and retrieve the command line output."
    args_schema: Optional[Type[BaseModel]] = PythonCodeInterpreterArgs
    interpreter: CodeInterpreter = Field(default_factory=CodeInterpreter)


    def _run(self, code: AnyStr) -> Any:
        return self.interpreter.execute_code(code)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


if __name__ == "__main__":
    tool = PythonCodeInterpreter()
    ans = tool._run("import os\nprint(\"helloworld\")")
    print(ans)
