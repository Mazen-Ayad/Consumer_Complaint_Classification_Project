# Consumer Complaint Classification (RNN vs. Transformer)

Reads a customer's written complaint and routes it to one of **5 categories**, using an end-to-end NLP pipeline: SimpleRNN, LSTM and GRU built from scratch in Keras, a fine-tuned **DistilBERT** (HuggingFace), a TF-IDF + Logistic Regression baseline, and a **Gradio** app for the final model.

Built as part of a Sprint Three (RNN & NLP) course project.

**Categories:** `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`

## Results (test set, 18,625 complaints)

| Model | Accuracy | Macro-F1 | Training time | Parameters |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | **0.8628** | **0.8507** | ~1.3 min (CPU) | n/a |
| SimpleRNN | 0.8154 | 0.7962 | 2.8 min | 2.57 M |
| LSTM | 0.8306 | 0.8172 | 1.2 min | 2.61 M |
| GRU | 0.8453 | 0.8342 | 0.9 min | 2.60 M |
| DistilBERT (fine-tuned) | 0.8479 | 0.8382 | 38.7 min | 67.0 M |

GPU times are on Kaggle (Tesla T4). Macro-F1 is the main metric because the classes are imbalanced (`credit_reporting` is ~45% of the data, `retail_banking` ~11%). Results vary by roughly ±0.005 between runs because of GPU non-determinism.

### What the results say

- **The simple baseline won.** TF-IDF + Logistic Regression beat every neural model. The text in this dataset was already reduced to keywords (stop words removed), which suits a bag-of-words model well.
- **GRU > LSTM > SimpleRNN.** The gated models handle long complaints better than a plain RNN.
- **DistilBERT is only +0.4 points of macro-F1 over GRU**, at about 43x the training time and 26x the parameters. Whether that is worth it depends on the use case.
- **Padding direction matters for SimpleRNN.** Pre-padding (zeros at the start) lets the final hidden state see real words; post-padding makes it read a long run of zeros last.
- **Every neural model overfits after 2-3 epochs** (train accuracy ~0.89-0.90 vs. validation ~0.83-0.84), so early stopping restores the best epoch.

## Pipeline

1. **Cleaning (before splitting, to avoid leakage):** dropped the old index column (it hides duplicates), 10 empty narratives, exact duplicates, 172 narratives that appear with conflicting labels, and very short texts. 162,421 rows became **124,164** (23.6% removed).
2. **Split:** stratified 70 / 15 / 15 with a fixed seed (86,914 / 18,625 / 18,625). No text is shared between splits.
3. **Tokenization:** vocabulary of 20,000 words built from the training set only (99.7% token coverage), sequences padded/truncated to 250 tokens.
4. **Imbalance:** class weights (`balanced`) in every model; macro-F1 for evaluation.
5. **Models:** `Embedding(128) -> recurrent layer(64) -> Dropout(0.3) -> Dense(5)` for the three RNNs (only the recurrent layer changes); `distilbert-base-uncased` fine-tuned for 2 epochs with a class-weighted loss.
6. **Evaluation:** accuracy, macro precision / recall / F1, per-class report and confusion matrices.

## Repository structure

```
app.py                    # Gradio app (DistilBERT)
distilbert_complaints/    # config + tokenizer (weights file not included, see below)
notebooks/
  step-1-data-cleaning-split.ipynb
  step-2-tokenizer-padding.ipynb
  step-3-baseline-simplernn.ipynb
  step-4-lstm-gru.ipynb
  step-5-distilbert.ipynb
requirements.txt
```

The notebooks were run on Kaggle in order; each step reads the previous step's output (`splits/`, `processed/`, `results/`) as a Kaggle *Notebook Output* input.

## Data

Public CFPB consumer complaint data, as the Kaggle dataset `shashwatwork/consume-complaints-dataset-fo-nlp` (`complaints_processed.csv`, 162,421 rows). The dataset is not included in this repo. The text is already lowercased, lemmatized and stripped of punctuation, digits and stop words (so negations such as "not" are gone too).

## Run the app

The fine-tuned weights (`model.safetensors`, ~255 MB) are larger than GitHub allows, so they are not in this repo. To get them, run `notebooks/step-5-distilbert.ipynb` (about 40 minutes on a Kaggle GPU) and copy the generated `model.safetensors` into the `distilbert_complaints/` folder here.

```bash
git clone <this repo's URL>
cd <repo folder>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: lemmatizer used to clean user input the same way as the training data
python -c "import nltk; nltk.download('wordnet')"

python app.py                      # http://127.0.0.1:7860
```

Optional environment variables: `MODEL_DIR` (model folder), `GRADIO_SHARE=1` (public link), `FEEDBACK_PATH` (where corrections are saved).

### App features

- Predicted category, confidence score and the team it would be routed to
- Auto-route above a confidence threshold (default 60%); below it, the complaint is flagged for a human agent
- Explanation: the words that pushed the model toward its prediction
- Batch mode: upload a CSV, download the predictions
- Session history and a feedback form (corrections saved to a local CSV)

## Limitations

- The model always picks one of the 5 categories, even for unrelated text; the confidence threshold is the safeguard.
- `credit_card` and `debt_collection` are the hardest classes (F1 around 0.79-0.80 for DistilBERT).
- The split is random, not by date, so the scores may be optimistic for complaints from a different time period.
- The routing table (team names) in the app is illustrative.

## Tech stack

Python, TensorFlow/Keras, PyTorch, HuggingFace Transformers, scikit-learn, Gradio, NLTK
