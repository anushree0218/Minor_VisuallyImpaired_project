import openai

class LLMIntegration:
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def generate_description(self, prompt):
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()