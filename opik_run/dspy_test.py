import os
import dspy
from opik.integrations.dspy.callback import OpikCallback


# Fix the model name and API configuration
lm = dspy.LM(
    "meta/llama-3.1-405b-instruct",  # Use a valid NVIDIA model
    api_base="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model_type="chat",
)

project_name = "DSPY"
opik_callback = OpikCallback(project_name=project_name, log_graph=True)

# Configure with both LM and callback
dspy.configure(lm=lm, callbacks=[opik_callback])


# Create a more specific signature
class QA(dspy.Signature):
    """Answer questions thoughtfully and clearly."""

    question = dspy.InputField()
    answer = dspy.OutputField()


rag = dspy.ChainOfThought("context, question -> answer")


# cot = dspy.ChainOfThought(QA)

# # Test the configuration
# try:
#     answer = cot(question="What is the meaning of life?")
#     print("Answer:", answer.answer)
# except Exception as e:
#     print(f"Error: {e}")
#     # Test basic LM connection
#     try:
#         response = lm("Hello, how are you?")
#         print("Direct LM response:", response)
#     except Exception as e2:
#         print(f"LM Error: {e2}")


r = rag(
    context=" You are in the north pole during November of the month.",
    question="You are close to which countries, give list of 3 closest countries? What is the typical temperature in your location?",
)


print("RAG Response:", r.answer)
print(r.inputs)
# print("RAG Full Response:", r)

r = rag(
    context=" I want to describe the characteristics of driver(trainer) and the traine(coach) in a driving school setting. There are possible violation involved with a driver/trainee such as not being alert, hands off the steering wheel, distracted. There are possible violations involved with a coach such as not paying attention to the road, use of electronic devices, eyes-close/tired or dozing off. There are possible violations involved with both driver and coach such as not wearing seatbelt, eating/drinking while driving, not following traffic rules, not following road signs, not following speed limits, not following lane discipline, not following safe distance, not following overtaking rules, not following parking rules.",
    question="What are the effective VLM prompts for describing the video content covering the violations aspects happening inside the car? ",
)
print("RAG Response:", r.answer)
print(r.inputs)
