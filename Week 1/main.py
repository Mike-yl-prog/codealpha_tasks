# ============================================================
# IMPORTS
# ============================================================

import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optin
from torch.utils.data import DataLoader, TensorDataset



# ============================================================
# NLTK DATA
# ============================================================
# These packages provide the natural-language-processing tools
# that our chatbot needs.
#
# punkt / punkt_tab:
#     Used by NLTK to split text into individual words.
#
# wordnet:
#     Used by the lemmatizer to reduce words to their base form.
#
# Example:
#
#     "running" -> "running"
#     "cars"    -> "car"
#
# This makes it easier for the model to recognize similar words.
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download("wordnet")




# ============================================================
# NEURAL NETWORK MODEL
# ============================================================

"""
    Neural network used to classify a user's message into
    one of the chatbot's known intents.

    Example intents:

        greeting
        goodbye
        order_tracking
        payment
        help

    The model receives a numerical representation of a sentence
    and produces a score for every possible intent.
    """
class chatbotModel(nn.Module):
    
    def __init__(self, input_size, output_size):
          # Initialize the parent PyTorch neural-network class.
        super(chatbotModel, self).__init__()

         # First fully-connected layer.
        #
        # input_size:
        #     Number of features in our input.
        #
        # 128:
        #     Number of neurons in this hidden layer.
        self.fc1 = nn.Linear(input_size, 128)
         # Second fully-connected layer.
        #
        # The previous layer has 128 outputs, so this layer
        # must receive 128 inputs

        self.fc2 = nn.Linear(128, 64)
        # Final layer.
        #
        # 64 inputs -> output_size outputs.
        #
        # output_size is the number of intents.

        self.fc3 = nn.Linear(64, output_size)
         # ReLU introduces non-linearity into the network.
        #
        # Without non-linear activation functions, stacking
        # linear layers would still behave like one linear
        # transformation.
        self.relu = nn.ReLU()
         # Dropout randomly disables some neurons during training.
        #
        # This helps reduce overfitting.
        self.dropout = nn.Dropout(0.0)

#defines hoe sata moves through the neural network.
# this is the forward pass
    def forward(self, x):
        # Input -> first layer -> ReLU -> Dropout
        x =  self.relu(self.fc1(x))
        x = self.dropout(x)

      # First hidden layer -> second layer -> ReLU -> Dropout
        x = self.relu (self.fc2(x))
        x = self.dropout(x)
     # Final layer produces one score for each intent.

        x = self.fc3(x)

        return x

 # ============================================================
# CHATBOT ASSISTANT
# ============================================================
class chatbotAssistant:
      # Main class responsible for preparing the chatbot's data,
    #training the neural network, saving/loading the model,
    # and processing user messages. #

      def __init__(self, intent_path, function_mapping=None):  

             # The trained neural network will eventually be stored here.
            self.model = None

             # Location of the JSON file containing our intents.
            self.intent_path = intent_path

            # Stores individual training examples.
        #
        # Example:
        #
        # [
        #     (["hello"], "greeting"),
        #     (["how", "are", "you"], "greeting")
        # ]
            self.documents = []
             # Contains every unique word found in the training data.
            self.vocabulary = []
            # Contains the names of all intents.
        #
        # Example:
        #
        # ["greeting", "goodbye", "help"]

            self.intents = []
        # Maps each intent to possible responses.
        #
        # Example:
        #
        # {
        #     "greeting": ["Hello!", "Hi!"],
        #     "goodbye": ["See you later!"]
        # }

            self.intents_responses = {}
             # Optional functions associated with specific intents.
        #
        # For example:
        #
        # "weather" -> get_weather()
        #
        # This allows an intent to trigger actual application logic.

            self.function_mappings = function_mapping

        # Training features.
            self.X = None

           # Training labels.   
            self.y = None

# ========================================================
    # TOKENIZATION + LEMMATIZATION
    # ========================================================
      @staticmethod
      def tokenize_and_lemmatize(text):

       # Converts a sentence into normalized words.
       # Example:
       #Where are my cars?"
       # might become:
       # ["where", "are", "my", "car"]
           lemmatizer =  nltk.WordNetLemmatizer()
       # reducing word into a stem word actually
           words = nltk.word_tokenize(text)

       # Convert words to lowercase and lemmatize them.
           words = [lemmatizer.lemmatize(word.lower())
               for word in words]

           return words

        # create chatbot
           chatbot =   chatbotAssistant('intent.json')


# turn it to a numerical
      
      def bag_of_words(self, words, vocabulary):
            return [ 1 if word in words else 0 for word in  self.vocabulary]


 # ========================================================
    # PARSE INTENTS
    # ========================================================
      def parse_intents(self): 
     #Reads the intents JSON file and extracts:

    #1. Intent names
    #2. Training patterns
    #3. Responses
    # 4. Vocabulary
    

          lemmatizer = nltk.WordNetLemmatizer()

    # Make sure the JSON file actually exists.
          if os.path.exists(self.intent_path):
           with  open(self.intent_path, 'r', encoding = "utf-8") as f:
            intents_data = json.load(f)

           for intent in intents_data['intents']:
             if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

              # Process every example sentence associated
              # with this intent.
             for pattern in intent['patterns']:
                 pattern_words = self.tokenize_and_lemmatize(pattern)
                 self.vocabulary.extend(pattern_words)
                 self.documents.append((pattern_words,intent['tag']))


        # Remove duplicate words and sort the vocabulary.
           self.vocabulary = sorted(set(self.vocabulary))



 # ========================================================
    # PREPARE TRAINING DATA
    # ========================================================
      def prepare_data(self):
          bags = []
          indices = []

    
       
    # Process every training example.   
          for document in self.documents:
           word = document[0]

    # Convert words into a bag-of-words vector. 

          for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words, self.vocabulary)

            intent_index = self.intents.index(document[1])


 # Store the numerical input.
            bags.append(bag)
            indices.append(intent_index)

   # Convert Python lists into NumPy arrays.
            self.X = np.array(bags)
            self.y = np.array(indices)
# ========================================================
    # TRAIN MODEL
    # ========================================================


      def train_model(self, batch_size, lr, epochs):

    # Convert input data from NumPy into PyTorch tensors.
        #
        # float32 is appropriate for neural-network input data.

           X_tensor = torch.tensor(self.X, dtype= torch.float32)
           Y_tensor = torch.tensor(self.y,  dtype= torch.long)

     # Combine inputs and labels into a PyTorch dataset.

           dataset = TensorDataset(X_tensor, Y_tensor)
           loader = DataLoader(
               dataset,
               batch_size=batch_size, 
               shuffle=True
               )


     # DataLoader divides the dataset into batches and
     # optionally shuffles the examples.


     # Create the neural network.
        #
        # Number of input features =
        # size of our vocabulary.
        #
        # Number of output classes =
        # number of intents.

           self.model = chatbotModel(self.X.shape[1], len(self.intents))


     # CrossEntropyLoss is commonly used for
        # multi-class classification.

           criterion = nn.CrossEntropyLoss()
           optimizer = optin.Adam(
               self.model.parameters(), lr=lr
               )


    # Adam updates the neural-network parameters
    # during training.
    

     # Training loop.
           for epoch in  range(epochs):
             
             running_loss = 0.0

     # Process one batch at a time.
             for batch_x, batch_y in loader:

              # Remove gradients from the previous iteration.
               optimizer.zero_grad()

            # Ask the model to make predictions.
               outputs = self.model(batch_x)

            # Compare predictions with the correct labels.
               loss = criterion (outputs, batch_y)


             # Calculate gradients.
               loss.backward()


           # Update the model's parameters.
               optimizer.step()

# Calculate average loss for this epoch.
               running_loss += loss

           print(
               f" {epoch + 1}: loss: {(running_loss / len(loader)):.4f}"
               )


# ========================================================
    # SAVE MODEL
    # ========================================================
      def save_model(self, model_path, dimensions_path):

    # Save the model's learned parameters.
         torch.save(self.model.state_dict(), model_path)

        # Save information about the model's dimensions.
        #
        # We need this information when loading the model.
         with open(dimensions_path, 'w') as f:
          json.dump({'input_size': self.X.shape[1], 'outpur_size': len(self.intents)}, f)


# ========================================================
    # LOAD MODEL
    # ========================================================

      def load_model(self, model_path, dimensions_path):

    # Load the saved model dimensions.
         with open(dimensions_path, 'r') as f:
           dimesions = json.load(f)


     # Recreate the same neural-network architecture.
         self.model = chatbotModel(dimesions['input_size'].dimensions['output_size'])
         self.model.load_state_dict(torch.load(model_path, weights_only=True))


     # ========================================================
    # PROCESS USER MESSAGE
    # ========================================================
      def  process_message(self, input_message):
       words = self.tokenize_and_lemmatize(input_message)
       bag = self.bag_of_words(words, self.vocabulary)


     # Convert the user's sentence into normalized words.
       bag_tensor = torch.tensor ([bag], dtype= torch.float32)


    # Put the model into evaluation mode.
       self.model.eval()

       # We don't need gradients when making predictions.
       with torch.no_grad():
        predictions = self.model(bag_tensor)


    # Ask the neural network for predictions.
       prediction_class_index = torch.argmax(predictions, dim=1).item()
       predicted_intent = self.intents[prediction_class_index]

       print("predicted intent:", predicted_intent)


    # If this intent has an associated function,
        # execute that function.
       if self.function_mappings:
         if predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

      # Get the possible responses for this intent.
       if self.intents_responses[predicted_intent]:
       # Return a random response. 
        return random.choice(self.intents_responses[predicted_intent])       
       else:
               return None       



def get_stocks():
          stocks = ['APPL', 'Meta', 'NVDA', '65', 'MSFT']

          return random.sample(stocks, 3)

if __name__ == '__main__':
         assistant = chatbotAssistant('intent.json', function_mapping={'get_stocks'})
         assistant.parse_intents()
         assistant.prepare_data()
         assistant.train_model(32, lr =0.001, epochs=100)

         assistant.save_model('chatbot_model.pth', 'dimension.json')

         while True:
             message = input('Enter your message:')


             if message == '/quit':
               break

             print(assistant.process_message(message))
          

          

