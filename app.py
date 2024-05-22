import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

# Define functions for generating abstract, description, problem type, assumptions, and hypotheses
def generate_abstract(domain, sub_domain, title):
    prompt = ChatPromptTemplate.from_template(f"""
    You are provided with a title, domain, and sub-domain. Use these details to create a concise abstract for the user's problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Your task is to generate a 30-word abstract that accurately summarizes the problem described by the title, ensuring relevance to the given domain and sub-domain.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title})

def generate_description(domain, sub_domain, title, abstract):
    prompt = ChatPromptTemplate.from_template("""
    You are provided with a title, domain, sub-domain, and an abstract. Use these details to create a detailed description of the user's problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Abstract:
    ```{abstract}```

    Your task is to generate a comprehensive and detailed description of the problem described by the title and abstract. Ensure that the description elaborates on the key points mentioned in the abstract and is relevant to the given domain and sub-domain.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "abstract": abstract})

def determine_problem_type(domain, sub_domain, title, abstract):
    prompt = ChatPromptTemplate.from_template("""
    You are provided with a title, domain, sub-domain, and an abstract. Use these details to determine the problem-type for the user's problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Abstract:
    ```{abstract}```
    These are the Problem Category:
    - **Operational**: Problems related to internal processes or efficiency.
    - **Technical**: Problems related to technology or technical processes.
    - **Strategic**: Problems impacting long-term goals or strategic directions.
    - **Customer Experience**: Problems affecting customer satisfaction or engagement.
    - **Compliance**: Legal or regulatory issues needing resolution.
    Your task is to identify and specify the problem-type based on the information provided. Ensure that the problem-type accurately reflects the nature of the problem described by the title and abstract, and is relevant to the given domain and sub-domain.
    Just specify the problem type(1-2 words) accurately strictly in this format from the above 5 options.
    "Problem Type: ____________"
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "abstract": abstract})

def generate_assumptions(domain, sub_domain, title, problem_type, description):
    prompt = ChatPromptTemplate.from_template("""
    You are provided with a title, domain, sub-domain, and a detailed description of the problem. Use this information to generate assumptions for the user. Generate a maximum of 5 such assumptions.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Problem Type:
    ```{problem_type}```                                         

    Description:
    ```{description}```


    Your task is to infer potential assumptions based on the provided problem details. Assumptions are underlying beliefs or conditions assumed to be true. Ensure that the generated assumptions are relevant to the given domain, sub-domain, and problem description.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "problem_type": problem_type, "description": description})

def generate_hypotheses(domain, sub_domain, title, problem_type, description):
    prompt = ChatPromptTemplate.from_template("""
    You are provided with a title, domain, sub-domain, problem type, and a detailed description of the problem. Use this information to propose experiment hypotheses for solving the problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Problem Type:
    ```{problem_type}```

    Description:
    ```{description}```

    Your task is to propose experiment hypotheses based on the provided problem details. Experiment hypotheses are proposed strategies or approaches that can be tested to solve the problem. Ensure that the generated hypotheses are relevant to the given domain, sub-domain, problem type, and description.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "problem_type": problem_type, "description": description})


# Initialize session state variables
if "domain" not in st.session_state:
    st.session_state.domain = ""
if "sub_domain" not in st.session_state:
    st.session_state.sub_domain = ""
if "title" not in st.session_state:
    st.session_state.title = ""
if "abstract" not in st.session_state:
    st.session_state.abstract = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "problem_type" not in st.session_state:
    st.session_state.problem_type = ""
if "assumptions" not in st.session_state:
    st.session_state.assumptions = ""
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = ""

def main():
    st.title("Problem Solver App")
    st.sidebar.header("Input")

    # Input fields
    st.session_state.domain = st.sidebar.text_input("Domain", value=st.session_state.domain)
    st.session_state.sub_domain = st.sidebar.text_input("Sub-domain", value=st.session_state.sub_domain)
    st.session_state.title = st.sidebar.text_input("Title", value=st.session_state.title)

    if st.sidebar.button("Feed Data"):
        st.session_state.abstract = ""
        st.session_state.description = ""
        st.session_state.problem_type = ""
        st.session_state.assumptions = ""
        st.session_state.hypotheses = ""

    # Generate Abstract
    if st.button('Generate Abstract'):
        st.session_state.abstract = generate_abstract(st.session_state.domain, st.session_state.sub_domain, st.session_state.title)
    
    # Display Abstract
    if st.session_state.abstract:
        st.header("Abstract")
        st.write(st.session_state.abstract)

    # Generate Description
    if st.session_state.abstract and st.button('Generate Description'):
        st.session_state.description = generate_description(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.abstract)
    
    # Display Description
    if st.session_state.description:
        st.header("Description")
        st.write(st.session_state.description)

    # Determine Problem Type
    if st.session_state.description and st.button('Classify Problem Type'):
        st.session_state.problem_type = determine_problem_type(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.abstract)
    
    # Display Problem Type
    if st.session_state.problem_type:
        st.header("Problem Type")
        st.write(st.session_state.problem_type)

    # Generate Assumptions
    if st.session_state.problem_type and st.button('Generate Assumption'):
        st.session_state.assumptions = generate_assumptions(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.problem_type, st.session_state.description)
    
    # Display Assumptions
    if st.session_state.assumptions:
        st.header("Assumptions")
        st.write(st.session_state.assumptions)

    # Generate Hypotheses
    if st.session_state.assumptions and st.button('Generate Hypothesis'):
        st.session_state.hypotheses = generate_hypotheses(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.problem_type, st.session_state.description)
    
    # Display Hypotheses
    if st.session_state.hypotheses:
        st.header("Hypothesis")
        st.write(st.session_state.hypotheses)

if __name__ == "__main__":
    main()