def get_summary_prompt(filename: str) -> str:
    return f"""
    You are an expert academic researcher tasked with extracting key information from a provided academic paper (filename: {filename}). 
    
    Carefully read the document and extract the information to fulfill the required schema structure. 
    
    Guidelines for extraction:
    1. **Core Research Question**: Identify the main hypothesis or problem the authors are trying to solve. If it is not explicitly stated, infer it from the Introduction.
    2. **Methodology**: Summarize the experimental design or theoretical framework briefly. Avoid listing minute technical details unless they are the novel contribution of the paper.
    3. **Key Findings**: What are the 1-3 most significant results? Keep this concise.
    4. **Limitations & Future Work**: This is the most crucial section. Exert maximum effort to find explicitly stated limitations of the study, contradictory findings, or directions the authors suggest for future research. Look closely at the Discussion and Conclusion sections. 
    
    If the document does not appear to be a real academic paper or is fundamentally unreadable, populate the fields with 'N/A' and note the issue in the title field.
    """.strip()


def get_gaps_prompt(subject: str, combined_summaries: str) -> str:
    return f"""
    You are a senior Principal Investigator leading a research lab. You have been provided with summaries of several recent academic papers focusing on the topic of "{subject}".
    
    Read the summaries carefully. Your objective is to conduct a meta-analysis to synthesize the current state of the art and, critically, to uncover systemic **Research Gaps** that warrant new investigation. 
    
    Produce a comprehensive, cohesive report formatted in Markdown, structured exactly as follows:
    
    ## 1. State of the Field Synthesis
    - Provide a cohesive narrative of what is currently known and established based on these papers. 
    - Identify the dominant methodologies and common themes. Do not just list the papers sequentially; synthesize their contributions.
    
    ## 2. Critical Analysis of Research Gaps
    - **Unexplored Territories**: What specific questions or variables are consistently ignored or acknowledged as missing across multiple papers?
    - **Methodological Limitations**: Are there widespread flaws or limitations in how this topic is currently being studied? Is there a new methodology or technology that should be applied?
    - **Contradictions & Tensions**: Are there conflicting findings between the papers that need resolution?
    
    ## 3. Formulated Research Proposals
    Based strictly on the gaps identified above, propose 3 novel, highly specific research studies. For each proposal, provide:
    *   **Proposed Title**: A professional, academic title.
    *   **Targeted Gap**: Which specific gap from section 2 does this address?
    *   **Proposed Approach/Methodology**: A brief 2-3 sentence overview of how this study would be conducted.
    *   **Expected Impact**: Why is solving this gap important to the broader field?

    Here are the paper summaries to analyze:
    
    {combined_summaries}
    """.strip()
