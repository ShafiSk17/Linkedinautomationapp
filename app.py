import os
import sys
import streamlit as st

# Add the current directory to sys.path
sys.path.append('C:/Users/venka/Desktop/Linkedin')

from lyzr_automata.agents.agent_base import Agent
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from lyzr_automata.tasks.task_literals import InputType, OutputType
from lyzr_automata.tasks.task_base import Task
from lyzr_automata.tasks.util_tasks import summarize_task
from lyzr_automata.tools.prebuilt_tools import linkedin_image_text_post_tool

# Streamlit UI
st.title("Automated LinkedIn Post Pipeline")
st.write("Enter the necessary API keys and LinkedIn credentials to run the pipeline.")

open_ai_api_key = st.text_input("OpenAI API Key", type="password")
linkedin_owner = st.text_input("LinkedIn Owner")
linkedin_token = st.text_input("LinkedIn Token", type="password")

# Log container
log_container = st.container()

def run_automated_linkedin_post_pipeline(open_ai_api_key, linkedin_owner, linkedin_token):
    logs = []

    def log(message):
        logs.append(message)
        log_container.write(message)

    # Define the models and tools
    open_ai_model_text = OpenAIModel(
        api_key=open_ai_api_key,
        parameters={"model": "gpt-4-turbo-preview", "temperature": 0.2, "max_tokens": 1500},
    )
    open_ai_model_image = OpenAIModel(
        api_key=open_ai_api_key,
        parameters={"n": 1, "model": "dall-e-3"},
    )

    # Define agents
    content_researcher_agent = Agent(
        prompt_persona="You are an AI journalist good at using the provided data and write an engaging article",
        role="AI Journalist",
    )
    linkedin_content_writer_agent = Agent(
        prompt_persona="You write engaging linkedin posts with the provided input data",
        role="Linkedin Content Creator",
    )

    # Define a custom tool for LinkedIn posts
    linkedin_post_tool = linkedin_image_text_post_tool(owner=linkedin_owner, token=linkedin_token)

    # Define tasks
    search_task = Task(
        name="Search Latest AI News",
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=open_ai_model_text,
        instructions="Search and collect all latest news about the startup Perplexity",
        log_output=True,
    )
    research_task = Task(
        name="Draft Content Creator",
        agent=content_researcher_agent,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=open_ai_model_text,
        instructions="Analyze the input and clean the data and write a summary of 1000 words which can be used to create Linkedin post in the next task",
        enhance_prompt=False,
    )
    linkedin_content_writing_task = Task(
        name="Linkedin Post Creator",
        agent=linkedin_content_writer_agent,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=open_ai_model_text,
        instructions="Use the news summary provided and write 1 engaging linkedin post of 200 words",
        log_output=True,
        enhance_prompt=False,
    )
    image_creation_task = Task(
        name="linkedin image creation",
        output_type=OutputType.IMAGE,
        input_type=InputType.TEXT,
        model=open_ai_model_image,
        log_output=True,
        instructions="Use the research material provided and create a linkedin post image that would be suitable for posting",
    )
    linkedin_upload_task = Task(
        name="upload post to linkedin",
        model=open_ai_model_text,
        tool=linkedin_post_tool,
        instructions="Post on Linkedin",
        input_tasks=[linkedin_content_writing_task, image_creation_task],
    )

    # Define and run the pipeline
    pipeline = LinearSyncPipeline(
        name="Automated Linkedin Post",
        completion_message="Posted Successfully 🎉",
        tasks=[
            search_task,
            research_task,
            linkedin_content_writing_task,
            summarize_task(15, text_ai_model=open_ai_model_text),
            image_creation_task,
            linkedin_upload_task,
        ],
    )

    # Capture logs during pipeline execution
    original_stdout_write = sys.stdout.write
    sys.stdout.write = log

    try:
        pipeline.run()
    finally:
        sys.stdout.write = original_stdout_write

    return logs

if st.button("Run Pipeline"):
    if not (open_ai_api_key and linkedin_owner and linkedin_token):
        st.error("Please provide all the required API keys and LinkedIn credentials.")
    else:
        logs = run_automated_linkedin_post_pipeline(open_ai_api_key, linkedin_owner, linkedin_token)
        st.success("Pipeline completed!")
        for log_entry in logs:
            log_container.write(log_entry)
