# Customer Review Sentiment Analyzer 🤖

A web application that uses a Scikit-learn NLP model to classify customer reviews as either positive or negative, served via a Flask API.

## Screenshot


*(To add your screenshot, drag and drop an image of your running application here in the GitHub editor.)*

---

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas
- **Frontend:** HTML, CSS, JavaScript

---

## Setup & Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd Customer-Review-Sentiment-Analyzer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

5.  Open your browser and go to `http://127.0.0.1:5000`.

---

## Model Overview
The model is a **Logistic Regression** classifier trained on the IMDb 50k Movie Reviews dataset. Text is vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which includes preprocessing steps like lowercasing, tokenization, and stopword removal. The model achieves an accuracy of approximately 88% on the validation set.
