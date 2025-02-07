import pandas as kl
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load dataset
data = kl.read_csv("chatbot_dataset.csv")

# Preprocess the data
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(x_train, y_train)
print("Model training complete")

# Function to get a response
def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Chatbot", style={'textAlign': 'center', 'color': 'black'}),
    dcc.Textarea(id='user-input', value="How can I help you?", style={'width': '100%', 'height': 100}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
], style={'backgroundColor': '#7370d0', 'padding': '20px', 'borderRadius': '10px', 'maxWidth': '500px', 'margin': 'auto'})

# Callback function to update chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks > 0:
        response = get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P(f"Bot: {response}", style={'margin': '10px', 'backgroundColor': '#f4f4f4', 'padding': '10px'})
        ])
    return False

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
