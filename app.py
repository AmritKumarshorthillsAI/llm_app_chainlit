# import chainlit as cl


# @cl.on_message
# async def main(message: cl.Message):
#     # Your custom logic goes here...

#     # Send a response back to the user
#     await cl.Message(
#         content=f"Received: {message.content}",
#     ).send()





import chainlit as cl
from src.llm import ask_order, messages

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    messages.append({"role": "user", "content": message.content})
    response = ask_order(messages) # extracting content
    messages.append({"role": "assistant", "content": response})

    # Send a response back to the chainlit user interface
    await cl.Message(
        content=response,
    ).send()


