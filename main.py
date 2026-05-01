from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import sympy as sp
from langchain_core.tools import tool

load_dotenv()

@tool
def calculator(a: float, b: float) -> float:
    """Useful for when you need to perform calculations. Input should be in the format: 'a operator b', e.g. '2 + 2'."""
    print("tool has been called")
    return f"{a} + {b} = {a + b}\n{a} - {b} = {a - b}\n{a} * {b} = {a * b}\n{a} / {b} = {a / b if b != 0 else 'undefined'}"

@tool
def integration(expression: str, variable: str = "x", lower: float = None, upper: float = None) -> str:
    """computes the integral of the given expression with respect to the specified variable. Optionally, you can provide lower and upper limits for definite integration."""
    try:
        var = sp.symbols(variable)
        expr = sp.sympify(expression)
        
        if lower is not None and upper is not None:
            result = sp.integrate(expr, (var, lower, upper))
            return f"The definite integral of {expression} with respect to {variable} from {lower} to {upper} is: {result}"
        else:
            result = sp.integrate(expr, var)
            return f"The indefinite integral of {expression} with respect to {variable} is: {result} + C"
    except Exception as e:
        return f"Error in symbolic calculation: {str(e)}"
    
@tool
def differentiation(expression: str, variable: str = "x") -> str:
    """computes the derivative of the given expression with respect to the specified variable."""
    try:
        var = sp.symbols(variable)
        expr = sp.sympify(expression)
        result = sp.diff(expr, var)
        return f"The derivative of {expression} with respect to {variable} is: {result}"
    except Exception as e:
        return f"Error in symbolic calculation: {str(e)}"
    
@tool
def solve_equation(equation: str, variable: str = "x") -> str:
    """solves the given equation for the specified variable."""
    try:
        var = sp.symbols(variable)
        eq = sp.sympify(equation)
        result = sp.solve(eq, var)
        return f"The solution(s) to the equation {equation} for variable {variable} is/are: {result}"
    except Exception as e:
        return f"Error in symbolic calculation: {str(e)}"

@tool    
def solve_thevenin_bridge(v_source: float, r1: float, r2: float, r_load: float) -> str:
    """Solves the Thevenin equivalent circuit for a bridge circuit."""
    try:
        # Simplified Thevenin equivalent calculation (replace with actual circuit analysis if needed)
        v_thevenin = v_source * (r2 / (r1 + r2))
        r_thevenin = (r1 * r2) / (r1 + r2)
        return f"The Thevenin equivalent voltage is: {v_thevenin}, and the Thevenin equivalent resistance is: {r_thevenin}"
    except Exception as e:
        return f"Error in Thevenin bridge calculation: {str(e)}"
    

def main():
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    tools = [calculator, integration, differentiation, solve_equation, solve_thevenin_bridge]

    # Prebuilt ReAct agent
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me anything, and I'll use Groq to answer for free.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break
        
        print("\nAssistant: ", end="", flush=True)
        
        # Streaming the response to the terminal
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            # In LangGraph, the 'agent' key contains the LLM's response
            if "agent" in chunk:
                msg = chunk["agent"]["messages"][-1]
                # Only print if it's a message with content
                if msg.content:
                    print(msg.content, end="", flush=True)
        
        print() # New line after response

if __name__ == "__main__":
    main()
