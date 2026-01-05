from flask import Flask, jsonify, render_template, request

from lstm import LongShortTermRNN
from utils import one_hot

text = "She opened the window and watched the rain fall gently on the streets below."
words = list(set(text.split()))
word_to_index = {w: i for i, w in enumerate(words)}
index_to_word = {i: w for w, i in word_to_index.items()}

seq_length = 2
data = []
tokens = text.split()
for i in range(len(tokens) - seq_length):
    seq = tokens[i : i + seq_length]
    target = tokens[i + seq_length]
    data.append(
        ([one_hot(word_to_index[w], len(words)) for w in seq], word_to_index[target])
    )

lstm = LongShortTermRNN(
    input_size=len(words), hidden_size=16, output_size=len(words), lr=0.1
)
for epoch in range(1000):
    for x_seq, y_idx in data:
        lstm.train(x_seq, y_idx)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    seq = content.get("sequence", [])
    x_seq = []
    for w in seq[-seq_length:]:  # take last seq_length words
        if w in word_to_index:
            x_seq.append(one_hot(word_to_index[w], len(words)))
    if not x_seq:
        return jsonify({"predicted_word": ""})
    pred = lstm.forward(x_seq)
    pred_word = index_to_word[pred.index(max(pred))]
    return jsonify({"predicted_word": pred_word})


if __name__ == "__main__":
    app.run(debug=True)
