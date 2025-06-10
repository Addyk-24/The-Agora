---
title: The Agora
emoji: 🌍
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: true
license: mit
short_description: Where artificial minds gather to forge wisdom
---

# The Agora: Where artificial minds gather to forge wisdom

### TRACK : mcp-server-track

## 🌟 Project Overview
Agora, also known as "AI Democracy," is an innovative Gradio-based server designed to foster collaborative decision-making among diverse large language models (LLMs).

Imagine an "AI Council" where specialized AI agents deliberate and vote on complex problems, providing reasoned arguments, highlighting disagreements, and ultimately arriving at a synthesized consensus.

This system transcends the limitations of single-model outputs by leveraging the unique strengths of various LLMs, making it perfect for scenarios demanding nuanced.

## ✨ Features
Multi-Model AI Council: Orchestrates a diverse panel of AI models, each playing a specific role:

- *Anthropic Claude: Specialized in ethical considerations and moral reasoning.*

- *OpenAI GPT (e.g., GPT-4o): Excels in creative problem-solving and brainstorming novel solutions.*

- *Mistral: Focused on robust technical analysis and detailed breakdowns.*

- *Sambanova: Provides rapid, high-throughput inference and quick factual recall.*

- *Hyperbolic Labs (placeholder for specialized models): Integrated for highly specialized tasks or domain-specific knowledge.*

- *Orchestrated AI Debates: 
Facilitates structured dialogues and 'debates' between AI models, allowing them to present arguments and counter-arguments.*

- *Transparent Reasoning: Each model's individual reasoning, thought process, and initial stance are transparently displayed.*

- *Disagreement Highlight: Clearly identifies areas of disagreement between models, providing insights into differing perspectives.*

- *Final Consensus & Synthesis: Synthesizes the collective insights and votes into a consolidated, consensus-driven final answer.*

- *Gradio User Interface: Provides an intuitive and interactive web interface for users to submit problems and view the council's deliberations.*

## 🚀 Workflow: 

- *How Agora Reaches Consensus:*
    - *Agora operates through a sophisticated, multi-stage process to transform a complex problem into a collective AI consensus.*
    - *The system acts as a Multi-Council Orchestration Protocol (MCP) server, managing the flow between the user interface and the various AI models.*

Here's a conceptual workflow:


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/uXoZKS0pFbKs6QZ97hw43.png)



- *User Problem Submission (Gradio UI):*

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/yD1v3mpJiEK8XmHnp2K21.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/2H9lYmYsgojkYJTbnMhXM.png)

IMAGE

A user submits a complex problem or query via the Gradio web interface. The input is typically a natural language prompt, potentially with accompanying data.


Image Description:

A screenshot of a Gradio interface with an input text box for the user's problem and a "Submit" button.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/13TAbpbWfpdvEnkDrc3LV.png)


- *Problem Parsing & Initial Distribution (MCP Orchestrator):*

- *The MCP Orchestrator (a custom backend server) receives the user's problem.*

- *It parses the input and determines the initial context for the AI Council.*

Based on pre-defined roles, the orchestrator dispatches the problem to specific models or groups of models for initial analysis and proposals. For instance, Claude might get an ethical framing, GPT a creative angle, and Mistral a technical breakdown.


- *A diagram showing the MCP Orchestrator sending the problem to multiple distinct AI models.*


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/xvbQmKjjqWB5RuPNm9foL.png)

- *Individual Model Reasoning & Proposals:*

  - *Each designated AI model processes the problem based on its specialty.*

  - *Models generate their initial solutions, ethical considerations, technical analyses, or creative approaches.*

  - *These individual outputs (including their 'reasoning' and 'confidence scores' if applicable) are sent back to the MCP Orchestrator.*



Debate Orchestration (MCP Orchestrator): Everything happens at backend and Final winner response is displayed in frontend

The orchestrator initiates a multi-turn 'debate' or 'review' phase.

- *Round 1 (Initial Review): Each model's proposal is shared (anonymously or attributed) with other relevant models.*
- *Round 2 (Rebuttal & Refinement): Models respond to critiques, refine their initial proposals, or adjust their positions.*

Image Description: A visual representation of AI models exchanging arguments, possibly with arrows indicating flow of information and feedback loops.

- *Voting & Consensus Formation:*

    - *After the debate rounds, the orchestrator prompts each AI model to "vote" on the most optimal solution or to provide a final, refined recommendation.*

- *A consensus algorithm (e.g., majority vote, weighted average based on model confidence/role importance, or a final synthesis by a designated 'moderator' AI) is applied to derive the final collective decision. Disagreements are explicitly logged.*


Result Presentation (Gradio UI):

- *The MCP Orchestrator sends the complete deliberation log, including:-*

    - *Each model's initial reasoning.*

    - *Key arguments and counter-arguments during the debate.*

    - *Areas of significant disagreement.*

    - *The final, synthesized consensus or voted-upon solution.*

    - *Gradio renders this information to the user in a clear, structured, and interactive format.*

- *A Gradio output screen showing a structured summary of the AI council's deliberation and the final consensus.*


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/bZy7nuBKvn-PWkt6TMiGz.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b994567a8d64d6f1d2bdf0/4svi9AzsLov8KuQE0Q0w3.png)

## 🛠️ Technologies Used
Frontend: Gradio (for interactive web interface)

Backend: Custom Python MCP Orchestrator (Flask/FastAPI recommended for server implementation)

### AI Models (via APIs):

- *Anthropic Claude*

- *OpenAI GPT (e.g., GPT-4o)*

- *Mistral AI*

- *Sambanova (or similar, e.g., via Hugging Face Inference API)*

- *Hyperbolic Labs (or other specialized custom models/APIs)*

## 🎯 Potential Use Cases
- *Medical Diagnoses: AI council reviewing patient data, lab results, and symptoms to propose the most likely diagnosis, considering ethical implications, treatment creativity, and technical accuracy.*

- *Legal Advice: Analyzing case details, precedents, and laws to provide comprehensive legal advice, weighing ethical considerations and strategic options.*

- *Business Strategy: Developing complex business plans, marketing strategies, or investment decisions by leveraging creative, analytical, and ethical AI perspectives.*

- *Scientific Research: Formulating hypotheses, designing experiments, and interpreting results across various scientific disciplines.*

## ⚙️ Setup and Installation


1. Clone the repository:
```
git clone https://huggingface.co/spaces/Agents-MCP-Hackathon/TheAgora

cd .\TheAgora\
```

2. Install dependencies:

```
pip install -r requirements.txt
```
3. Run the MCP App:
```
python app.py
```

## 🤝 Contributing

Aditya Katkar\
[Github](https://github.com/Addyk-24)\
[LinkedIn](https://www.linkedin.com/in/aditya-katkar-673930340)