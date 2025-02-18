---
layout: page
title: "AI-Powered Flashcard Generator: A Case Study"
permalink: /case-studies/flashcard-generator/
---

### The Problem and Vision

Flashcards are useful but annoying.

They are a good way to memorize vocabulary and concepts, but the creation of flashcards takes away precious time that could be used for actual studying. Another issue is that students should have novelty when learning.

My goal with the Flashcard generator was to minimize the creation time and maximize the novelty for English as a Foreign language students.

Automating the process using generative AI like LLMs for text and diffusion models for images.

Ideally, this would allow students to create new flashcards quickly and reliably. They would be able to identify an unknown word, press a couple of buttons, and save them for future study sessions.

At least that was the goal.

Here is the basic layout of the system I wanted to create.

The student enters a word that they want to memorize. The first LLM call will define the word. Once confirmed the second LLM call will write a sample sentence with it. Then, that sample sentence is used as a prompt for the diffusion model to create an image.

The complete flashcard is saved for later study in a simple UI.

### Technical Foundation

The end goal was to create an Android app because most EFL students will have access to an Android phone before they have access to a laptop computer.

But, I wanted to create a prototype before investing the time into a full mobile app. I decided to go with Flask for two reasons. Flask is a simple stripped-down framework that would allow me to add API calls with straightforward server calls.

The second reason is a bit more nuanced. Claude, who I used for pair programming, is strongest with the Python language. Interpreted languages in general are easier for him rather than compiled languages like the C family or Go. 

I also needed a framework to handle multiple AI prompts for the different steps of the system.

CrewAI was picked because I had heard about it and was curious to try it. There wasn't much beyond that decision. I often select a new framework for personal projects for no other reason than I want to try it out.

For the AIs, I focused on simplicity and low cost. I wanted each flashcard to be made for the smallest amount of money possible. A student should be able to create and discard flashcards as they progress without feeling like they are throwing money away.

A frontier model like ChatGPT or Claude would be overkill. Like going grocery shopping in a Formula one car. All I needed was the AI to define a word and write a sentence. 

```
# Example of basic Flask setup with API integrations
from flask import Flask, render_template, request, jsonify
import groq
import replicate

app = Flask(__name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_KEY'))
```
*Setting up Flask web framework and connecting to our AI services. Flask handles web requests, Groq provides text generation, and Replicate manages image creation.*

Simple. 

So I decided on the Llama 3.3 3B model which costs $0.06 per million tokens. A word's definition is at most a couple dozen tokens, so calling this model is some vanishing fraction of a penny that is almost negligible.

The highest cost is still small when it came to the diffusion model. I went with the smallest image generative model I could find. It's the Flux Schnell model hosted on Replicate. It creates images for $0.003 each. 

That means a student could create a hundred flashcards for around $0.30. 

Problem and solution identified. 

Technology decided on.

Time for development.

### Development Journey

This was before the Model Context Protocol was released so I had to copy, paste, and alter Claude's code as we worked. 

Setting up the Python environment is always annoying on a Windows machine, but I installed Flask and CrewAI without much issue.

My workflow involved using VScode while communicating with Claude.

Once we got everything set up, I created a task list with Claude to break down the system into steps. The user journey, basically. The web development parts were easy because this was a prototype. The UI was a basic text box and some buttons. Nothing fancy.

When we started to set up the various API calls with CrewAI, we ran into an issue. At first, I was trying to write the prompts for the small LLMs. This was fine but slowed us down.

I discovered 'Meta-prompting' which is having a frontier model, like Claude writing prompts for me.

This was a wonderful change. Not only did it save time, but it also gave a much better outcome since Claude could understand how to speak with smaller models.

He could see the output, and adjust the prompts so we could iterate quickly. Once, we had the three basic prompts set up, we were getting the desired outputs for the most part.

But any journey worth taking has its challenges along the way.

The first major issue we encountered was the misunderstanding the 3B model was repeating the definitions instead of creating a definition and a sample sentence.

After some troubleshooting, Claude and I came up with a more robust error handling system that gave us the desired functionality. I learned about the limitations of these smaller models. They need a lot of hand holding and crystal clear guidance.

```
# Example of LLM interaction with error handling
def get_ai_content(word):
    try:
        prompt = f"""Define the word '{word}' in this exact format:
        Definition: (write a brief dictionary definition)
        Sentence: (write a different example sentence using the word)"""
        
        response = client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return definition, sentence
    except Exception as e:
        logger.error(f"Error getting AI content: {str(e)}")
        return "Error occurred", "Please try again!"
```
*Error handling system that ensures clear instruction formatting for the smaller LLM. If anything goes wrong, it gracefully returns an error message instead of breaking.*

At the root of an LLM, it is a statistical model at the end of the day. Imagine rolling dice, then imagine rolling hundreds and thousands of dice over and over. Errors are inevitable.

An important insight about the limitations of Large Language Models in general.

Now, we have a way for the user to enter a word, the first LLM call creates the definition, and the second call uses that definition to create a sample sentence.

It worked consistently with a negligible error rate. The user would be able to regenerate the definition or sentence with a quick press of the button.

```
-- Database schema showing card states and data storage
CREATE TABLE flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    definition TEXT NOT NULL,
    sample_sentence TEXT,
    image_data BLOB,
    status TEXT NOT NULL 
        CHECK (status IN (
            'definition_only',
            'has_sentence',
            'complete',
            'skipped_image'
        ))
);
```
*Database design that tracks each flashcard's journey from initial word to complete card with definition, sentence, and image. The status field ensures we know exactly where each card is in the creation process.*

The next part was the diffusion model. I had not worked with text-to-image models like this up to this point. I tried a local image model on my computer as an early project, but never used it for anything other than some random experimentation.

I learned that these image models work very different from LLMs. Their command of language is very limited since they can't engage in discussion like their text based cousins.

What I didn't know then was that these models need highly structured prompts because a slight bit of confusion will give you strange and sometimes disturbing image results.

Pressing on, Claude and I implemented the Replicate hosted Schnell model. We simply fed in the sample sentence to create a relevant image. 

It both worked and didn't work.

```
# Image generation and storage
@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        sentence = request.form['sentence']
        output = replicate_client.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": sentence}
        )
        image_url = str(output[0])
        return jsonify({
            'status': 'success',
            'image_url': image_url
        })
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate image'
        }), 500
```
*API endpoint that manages image generation requests. It takes a sentence, sends it to the Replicate API, and returns either a successful image URL or handles any errors that occur.*

### Key Insights About AI

So, yes, it created images consistently but it was hard to have any quality control over the output. A sample sentence might cause the diffusion model to create an image that was confusing or irrelevant. Then I learned another lesson.

Customer-facing generative AI models, where a random customer is pressing the buttons leave open the possibility of a negative user experience. 

AI in general is a powerful technology and without understanding how it works can lead to confusion at best and reputational damage at worst.

The importance of clear direction when dealing with small LLMs can't be understated. A 3B model is too small to infer what you want from vague directions. For context, Claude has 200B+ parameters and ChatGPT has over 1 trillion parameters.

Understanding the limitations of the model is critical when adding them to an application. Now, could we simply use a frontier-level model to avoid these miscommunications? Yes, but then we would be overspending to avoid being more precise.

When it comes to the image generation models, I've basically ignored them. I think they still need development and I need to better understand how to produce high-quality and reliable images.

I see the diffusion models as more of an internal tool to produce custom images, and eventually video, in-house then it can be moderated and edited as needed.

The key lesson with this project is to use caution when creating customer-facing AI-driven features.

### Lessons and Evolution

This was more of a learning opportunity than a commercially viable application. The idea for this had been on my mind for months. But when I had the final project in hand, I realized that there was an error in my thinking.

Systems design is more important than ever when dealing with AI in production. I technically made my idea a reality but I didn't think about the perspective of the user. 

If a user was trying to learn a new word, then how would they be able to validate the definition, sample sentence, or the final image? 

Only realizing this after I had built it made me grateful this was an inexpensive personal project. In the future, I need to challenge my design assumptions and view the system from the perspective of the user and from the AI model itself.

That last part might be confusing, to think about the AI's perspective, but it's important to the health and functionality of the entire system. Just like keeping in mind the user's perspective, understanding how AI models work is important as well.

In the next language learning app I am working on, Mountain ESL, I have focused on more structured management of the output to validate whether or not the information being taught is quality without the user needing to understand the system.

### Conclusion

AI is going to be a powerful force for the accessibility and quality of education but only if it is implemented with care and a clear idea of the outcome.

The idea that English language education could be available to anyone in the world for pennies per hour of study will be transformative for the billions of people needing to learn English to improve their lives around the world.

Reflecting on this project, I feel this was a pivotal moment for me as a developer. I saw AI's strengths and weaknesses first hand.

Claude could speed up my work as long as I knew the direction to go and the technical errors that we ran across. He can create solid code quicker than any human can move but needs a human to focus on the higher level thinking that might be outside of his current context window.

For me, I realized that even though I know how to work with Claude and other AI systems, I still need to work on the fundamentals of design, systems thinking, and debugging.

The Flashcard Generator will be mothballed now, but I'll carry the lessons I learned here into my next English learning application.

