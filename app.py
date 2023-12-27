# app.py (Flask server)

from flask import Flask, request, render_template, jsonify
from transformers import QuestionAnsweringPipeline

app = Flask(__name__)

# Load the question-answering model
model_name = "deepset/roberta-base-squad2"
qa_model = QuestionAnsweringPipeline(model=model_name, tokenizer=model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    question = request.form.get('question')

    if file and question:
        # Save the uploaded file (you might want to store it in a proper location)
        file.save('uploaded_file.txt')

        # Perform question answering on the uploaded file
        with open('uploaded_file.txt', 'r', encoding='utf-8') as file_content:
            context = file_content.read()

        answer = qa_model(question=question, context=context)

        return jsonify({"question": question, "answer": answer['answer']})

    return 'File upload or question missing'

if __name__ == '__main__':
    app.run(debug=True)
