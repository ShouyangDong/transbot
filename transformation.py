import openai

prompt_dict = {
    "loop_fuse": "Please the two for loops into one and update its corresponding loop index",
    "loop_split": "please split the for loop into two loop and update its corresponding loop index",
    "loop_reorder": "please reorder the two for loops",
    "loop_bind": "please bind the loop with corresponding parallel variables",
    "func_prefix": "sing the __global__ keyword to define a kernel function. A kernel function is a parallel function that runs on the GPU.",
}


def get_LLM_answers(question, action):
    prompt = f"{question}\n Transform the program above according to the given description: '{prompt_dict[action]}'"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        if "choices" in response and len(response["choices"]) > 0:
            message_content = response["choices"][0]["message"]["content"]
            print(f"Received answer: {message_content}")
            return [message_content]
        else:
            print("No valid response in 'choices'")
            return None
    except openai.error.OpenAIError as e:
        print(f"errrrrrrrorrr: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None