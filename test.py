from google import genai

client = genai.Client(api_key="AIzaSyBAuUFUfDoBkTsLE6YRbiZgeCHQjR6IOsg")

try:
    # Use the 2026 stable alias
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents="Hello! Is this the active model?"
    )
    print(" SUCCESS: Gemini 2.5 Flash is active!")
    print(response.text)
except Exception as e:
    print(f" STILL FAILING: {e}")