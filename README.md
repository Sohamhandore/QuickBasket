![image](https://github.com/user-attachments/assets/d0157117-3ad0-4d38-8f9d-942dc4b4171c)

![image](https://github.com/user-attachments/assets/8fdb0e38-39f0-4845-9a3a-cfe0febd1063)


---

# QuickBasket – AI Assistant for Customer Support

A streamlined conversational AI tool designed to assist customers in an e-commerce setting, leveraging locally hosted pre-trained models.

## 🔧 Key Capabilities

* Intent detection using DistilBERT
* Context-sensitive replies
* Streamlit-based minimalist chat interface
* Handles six core user intents:

  * Track My Order
  * Returns & Refunds
  * Check Product Availability
  * Store Timings & Location
  * Greetings & Small Talk
  * Unrecognized or Miscellaneous Queries

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
```

### 2. Install project dependencies

```bash
pip install -r requirements.txt
```

## 💬 How to Use

### 1. Launch the application

```bash
streamlit run app.py
```

### 2. Access the app in your browser

Typically available at: `http://localhost:8501`

### 3. Chat and explore its responses!

## 🧠 Under the Hood

* **Model**: DistilBERT (`distilbert-base-uncased`)
* Fine-tuned for intent classification tasks
* Response mechanism based on a rule-based system
* Context management for product-related interactions

## 💡 Example Interactions

### ➤ Order Inquiry

```
User: Where is my order?
Bot: Kindly provide your order ID so I can track it for you.
```

### ➤ Availability with Context Awareness

```
User: Do you have Nike shoes?
Bot: I can assist you with checking nike's availability. Could you specify the product?
```

### ➤ Store Hours

```
User: What time do you open?
Bot: We’re open from 9 AM to 9 PM, Monday through Saturday. Need help finding your nearest store?
```

## 🧰 Tech Stack

* `transformers`: for leveraging DistilBERT
* `torch`: for model operations
* `pandas`: for structured data manipulation
* `scikit-learn`: preprocessing and evaluation
* `streamlit`: UI for interaction
* `numpy`: numerical computations

## ⚠️ Known Limitations

* Supports only fixed set of intents
* Context memory is basic
* Lacks persistent chat history
* Model trained on limited data

## 🚧 Planned Enhancements

* Expand the intent dataset
* Enhance context tracking across multiple messages
* Integrate a product search engine
* Smarter response formulation
* Display confidence levels
* Enable support for multi-turn conversations

---

