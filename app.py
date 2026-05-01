import streamlit as st
import sympy as sp
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# 1. SETUP & CONFIGURATION
load_dotenv()
st.set_page_config(page_title="AI assistant", page_icon="⚡", layout="wide")

# 2. DEFINE ENGINEERING TOOLS
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
    



# 3. INITIALIZE AGENT
@st.cache_resource
def get_agent():
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [solve_thevenin_bridge, integration, differentiation, solve_equation, calculator]
    return create_react_agent(model, tools)

agent_executor = get_agent()

# 4. USER INTERFACE (UI)
st.title("⚡ Engineering AI Assistant")
st.markdown(f"""
Welcome to your engineering workspace. This agent can solve **Thevenin Theorems**, 
perform **Calculus**, and help with your engineering coursework.
""")

# Sidebar for Project Info (Branding)
with st.sidebar:
    st.header("Project Info")
    st.info("Built for Engineering Students")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask: 'What is the derivative of sin(x)*x?' or 'Solve Thevenin for 12V source...'"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the agent response
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=prompt)]}):
            if "agent" in chunk:
                msg = chunk["agent"]["messages"][-1]
                if msg.content:
                    full_response += msg.content
                    response_placeholder.markdown(full_response + "▌")
            elif "tools" in chunk:
                st.caption("🔍 Engineering Tool active...")
        
        response_placeholder.markdown(full_response)
    
    # Save Assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})