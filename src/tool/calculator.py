from typing import AnyStr
import re
import numexpr
from .basetool import *
import math


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="a mathematical expression.")


def _evaluate_expression(expression: str) -> str:
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
    except Exception as e:
        return f"numexpr.evaluate({expression.strip()}) raised error: {e}." \
            " Please try again with a valid numerical expression"

    # Remove any leading and trailing brackets from the output
    return re.sub(r"^\[|\]$", "", output)


class Calculator(BaseTool):
    """docstring for Calculator"""
    name = "calculator"
    description = "A calculator that can compute arithmetic expressions. Useful when you need to perform " \
                  "numerical calculations."
    args_schema: Optional[Type[BaseModel]] = CalculatorArgs

    def _run(self, expression: AnyStr) -> Any:

        response = _evaluate_expression(expression)
        evidence = response.strip()
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


if __name__ == "__main__":
    ans = Calculator()._run("1+1*213213")
    print(ans)
