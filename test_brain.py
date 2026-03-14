from brain.context_builder import build_context
from brain.prompt_manager import build_prompt
from brain.llm_engine import generate_response


def main():

    while True:

        user_input = input("User: ")

        context = build_context(user_input)

        prompt = build_prompt(context)

        response = generate_response(prompt)

        print("AI:", response)


if __name__ == "__main__":
    main()
