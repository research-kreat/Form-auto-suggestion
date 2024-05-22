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

    Your task is to identify and specify the problem-type based on the information provided. Ensure that the problem-type accurately reflects the nature of the problem described by the title and abstract, and is relevant to the given domain and sub-domain.
    Just specify the problem type(2-3 words) accurately strictly in this format.
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


# Streamlit app
def main():
    st.title("Problem Solver App")
    st.sidebar.header("Input")

    # Input fields
    domain = st.sidebar.text_input("Domain")
    sub_domain = st.sidebar.text_input("Sub-domain")
    title = st.sidebar.text_input("Title")

    if st.sidebar.button("Generate Solutions"):
        # Generate abstract
        abstract = generate_abstract(domain, sub_domain, title)
        st.header("Abstract")
        st.write(abstract)

        # Generate description
        description = generate_description(domain, sub_domain, title, abstract)
        st.header("Description")
        st.write(description)

        # Determine problem type
        problem_type = determine_problem_type(domain, sub_domain, title, abstract)
        st.header("Problem Type")
        st.write(problem_type)

        # Generate assumptions
        assumptions = generate_assumptions(domain, sub_domain, title, problem_type, description)
        st.header("Assumptions")
        st.write(assumptions)

        # Generate hypotheses
        hypotheses = generate_hypotheses(domain, sub_domain, title, problem_type, description)
        st.header("Hypotheses")
        st.write(hypotheses)

if __name__ == "__main__":
    main()
