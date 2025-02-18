**Sentiment Analysis Tool** 🌟
==============================

A powerful and user-friendly sentiment analysis tool built using Python. This tool utilizes machine learning and natural language processing (NLP) techniques to analyze and classify user comments into different sentiment categories, including **Positive**, **Neutral**, **Negative**, and **Aggressive**. With a clean and intuitive user interface powered by `Tkinter`, users can quickly input text and receive sentiment predictions.

* * *

**Key Features** ✨
------------------

*   **Real-Time Sentiment Prediction**: Get instant sentiment analysis for any text input.
*   **User-Friendly Interface**: A sleek and easy-to-use graphical user interface (GUI) built with Tkinter.
*   **Text Preprocessing**: Clean and preprocesses text by removing unwanted characters, URLs, mentions, and hashtags for accurate predictions.
*   **K-Nearest Neighbors (KNN)**: Uses KNN algorithm to classify sentiments based on pre-trained data.
*   **Visualizations**: Sentiment distribution visualizations based on the training data.

* * *

**Technologies Used** 🛠️
-------------------------

*   **Python 3.x**: The core programming language.
*   **Tkinter**: For creating the user interface.
*   **Scikit-learn**: For machine learning algorithms and preprocessing.
*   **Pandas**: For data manipulation and analysis.
*   **Seaborn & Matplotlib**: For data visualizations.
*   **NLP Techniques**: Text cleaning and vectorization using TfidfVectorizer.

* * *

**Installation** 🚀
-------------------

### 1\. Clone the repository:

bash

CopyEdit

`git clone https://github.com/ahmad-nadeem-official/sentiment_analysis.git
cd sentiment-analysis-tool` 

### 2\. Install required dependencies:

You can install the required packages using `pip`:

bash

CopyEdit

`pip install -r requirements.txt` 

* * *

**How to Run** 🚀
-----------------

1.  Make sure all dependencies are installed.
2.  To start the sentiment analysis tool, simply run the `ui.py` file:

bash

CopyEdit

`python ui.py` 

The GUI will open, allowing you to input any comment or text for sentiment prediction. The result will be displayed as either **Positive**, **Neutral**, **Negative**, or **Aggressive**.

* * *

**How it Works** 🤖
-------------------

1.  **Text Preprocessing**:
    
    *   All comments are cleaned to remove URLs, special characters, hashtags, and mentions.
    *   The text is converted to lowercase to ensure uniformity.
2.  **Sentiment Prediction**:
    
    *   A K-Nearest Neighbors (KNN) model is used to classify the sentiment of the comment based on pre-trained data.
    *   The training dataset consists of comments labeled with sentiments such as **Positive**, **Neutral**, **Negative**, and **Irrelevant**.
3.  **Model Evaluation**:
    
    *   The model is trained using `TfidfVectorizer` to convert the text into a numerical form.
    *   The sentiment labels are encoded, and the model predicts the sentiment of the provided input.
4.  **Real-Time UI**:
    
    *   Users can input text into the application, click on the "Predict Sentiment" button, and instantly receive a sentiment classification.

* * *

**Contributing** 🤝
-------------------

We welcome contributions from the community! If you'd like to improve the tool or fix any issues, please follow these steps:

1.  Fork this repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -am 'Add new feature'`).
4.  Push your changes (`git push origin feature/your-feature`).
5.  Open a Pull Request.

* * *

**Contact** 📬
--------------

*   **Ahmad**: \[ahmadnadeem095@gmail.com\]

* * *

**License** 📜
--------------

This project is licensed under the MIT License - see the LICENSE file for details.

* * *

**Acknowledgments** 💡
----------------------

*   Thanks to the contributors of the `scikit-learn`, `Tkinter`, and `Pandas` libraries for providing excellent tools for machine learning and data analysis.
*   Inspiration from various sentiment analysis projects.

* * *

### 💡 **Enjoy Predicting Sentiments with the Tool!** 💬