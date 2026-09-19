# BoundedGlitchEngine

A small trainable chatbot. No ML dependencies. Only Flask.

    pip install -r requirements.txt
    python app.py          # http://localhost:5000

## How it answers
1. **Taught answers**: `teach: question = answer` (saved to data/knowledge.json)
2. **Corpus search**: TF-IDF over .txt/.md files in data/corpus/
3. **Markov fallback**: word chain trained on the corpus + taught answers

## Commands
    teach: question = answer
    forget: question
    train        reload corpus files, rebuild
    stats
    help

## Layout
    model/       KnowledgeModel: storage, TF-IDF index, Markov chain
    inference/   generate(): picks the answer layer
    chatbot.py   BoundedGlitchEngine: commands + history
    app.py       Flask routes: / /chat /teach /train /stats
