dataset2prompt = {
    "versicode":
    {
        "vscc":"",
        "vace":""
    },
    "versiBCB":
    {
        "vscc":"",
        "vace":""
    }
}
dataset2prompt["versicode"]["vace"] = """
            You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code. 
            I will provide you with some content from the same dependency library for help.

            Important Notes:
            1. If the provided external knowledge (e.g., dependency documentation, code blocks, or version-specific details) is incomplete or missing, you should rely on your internal knowledge to infer the best approach for refactoring the code.
            2. If you encounter ambiguities or uncertainties due to missing external knowledge, clearly state any assumptions you are making to proceed with the refactoring.
            3. Your goal is to produce functional and optimized code that aligns with the new version of the dependencies, even if some external information is unavailable.

            

            ### Helpful code blocks
            {source_code}
            ### Functionality description of the code
            {description}

            ### Dependency and old version
            {original_version}

            ### Old version code
            {original_code}

            ### Dependency and new version
            {target_version}

            Please only return the refactored code and enclose it with `<start>` and `<end>`.

            ### Refactored new code
"""
dataset2prompt["versicode"]["vscc"] = """
            You are now a professional Python programming engineer. I will provide you with some content related to the dependency package. Then, I will also provide you with a functional description and specific dependency package version. 

            Your task is to write Python code that implements the described functionality using the specified dependency package and version.


            ### related content
            {knowledge_doc}

            Please only return the implementation code without any explanations. Enclose your code with `<start>` and `<end>` tags:

            ### Functionality description
            {description}

            ### Dependency and version
            {dependency}=={version}

"""