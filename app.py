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

def improve_abstract(domain, sub_domain, title, abstract, feedback):
    prompt = ChatPromptTemplate.from_template(f"""
    You are provided with a title, domain, sub-domain, and feedback on a previously generated abstract. Use these details to improve the abstract for the user's problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Abstract:
    ```{abstract}```

    Feedback:
    ```{feedback}```

    Your task is to generate a better abstract based on the feedback, ensuring relevance to the given domain and sub-domain.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "abstract": abstract, "feedback": feedback})


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

def improve_description(domain, sub_domain, title, abstract, feedback):
    prompt = ChatPromptTemplate.from_template(f"""
    You are provided with a title, domain, sub-domain, an abstract, and feedback on a previously generated description. Use these details to improve the description for the user's problem.

    Domain:
    ```{domain}```

    Sub-domain:
    ```{sub_domain}```

    Title:
    ```{title}```

    Abstract:
    ```{abstract}```

    Feedback:
    ```{feedback}```

    Your task is to generate a better description based on the feedback, ensuring relevance to the given domain and sub-domain.
    """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"domain": domain, "sub_domain": sub_domain, "title": title, "abstract": abstract, "feedback": feedback})


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
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "improved_abstract" not in st.session_state:
    st.session_state.improved_abstract = ""
if "generate_description_enabled" not in st.session_state:
    st.session_state.generate_description_enabled = False
if "dislike_clicked" not in st.session_state:
    st.session_state.dislike_clicked = False

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
        st.session_state.feedback = ""
        st.session_state.improved_abstract = ""
        st.session_state.generate_description_enabled = False
        st.session_state.dislike_clicked = False

    # Generate Abstract
    if st.button('Generate Abstract'):
        st.session_state.abstract = generate_abstract(st.session_state.domain, st.session_state.sub_domain, st.session_state.title)
        st.session_state.feedback = ""
        st.session_state.improved_abstract = ""
        st.session_state.dislike_clicked = False
    
    # Display Abstract
    if st.session_state.abstract:
        st.header("Abstract")
        st.write(st.session_state.abstract)

        if st.button("Like Abstract"):
            st.session_state.improved_abstract = st.session_state.abstract  # Mark the abstract as final
            st.session_state.generate_description_enabled = True
            st.session_state.dislike_clicked = False

        if st.button("Dislike Abstract"):
            st.session_state.dislike_clicked = True

        if st.session_state.dislike_clicked:
            st.session_state.feedback = st.text_input("What is missing in the abstract?", key="feedback_input")
            if st.session_state.feedback:
                if st.button("Improve Abstract", key="improve_abstract_button"):
                    st.session_state.improved_abstract = improve_abstract(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.abstract, st.session_state.feedback)
                    st.session_state.abstract = st.session_state.improved_abstract
                    st.session_state.feedback = ""  # Reset feedback
                    st.experimental_rerun()

    if st.session_state.improved_abstract:
        st.header("Improved Abstract")
        st.write(st.session_state.improved_abstract)

        if st.button("Like Improved Abstract"):
            st.session_state.generate_description_enabled = True

        if st.button("Dislike Improved Abstract"):
            st.session_state.dislike_clicked = True

        if st.session_state.dislike_clicked:
            st.session_state.feedback = st.text_input("What is missing in the improved abstract?", key="feedback_input_improved")
            if st.session_state.feedback:
                if st.button("Improve Abstract Again", key="improve_abstract_button_again"):
                    st.session_state.improved_abstract = improve_abstract(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.improved_abstract, st.session_state.feedback)
                    st.session_state.abstract = st.session_state.improved_abstract
                    st.session_state.feedback = ""  # Reset feedback
                    st.experimental_rerun()

    # Enable Description generation only if abstract is accepted
    if st.session_state.generate_description_enabled:
        if st.button('Generate Description'):
            st.session_state.description = generate_description(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.improved_abstract)
        
        # Display Description
        if st.session_state.description:
            st.header("Description")
            st.write(st.session_state.description)

        # Determine Problem Type
        if st.button('Classify Problem Type'):
            st.session_state.problem_type = determine_problem_type(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.improved_abstract)
        
        # Display Problem Type
        if st.session_state.problem_type:
            st.header("Problem Type")
            st.write(st.session_state.problem_type)

        # Generate Assumptions
        if st.button('Generate Assumption'):
            st.session_state.assumptions = generate_assumptions(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.problem_type, st.session_state.description)
        
        # Display Assumptions
        if st.session_state.assumptions:
            st.header("Assumptions")
            st.write(st.session_state.assumptions)

        # Generate Hypotheses
        if st.button('Generate Hypothesis'):
            st.session_state.hypotheses = generate_hypotheses(st.session_state.domain, st.session_state.sub_domain, st.session_state.title, st.session_state.problem_type, st.session_state.description)
        
        # Display Hypotheses
        if st.session_state.hypotheses:
            st.header("Hypotheses")
            st.write(st.session_state.hypotheses)

if __name__ == "__main__":
    main()
