# Import necessary modules
import re
import random
from iexfinance import Stock
import iexfinance
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
import spacy
from wxpy import *


############################################################
# Initialization part                                      #
############################################################

# Define the states
INIT = 0
AUTHED = 1
CHOOSE_STOCK = 2
CHOOSE_OTHER_STOCK = 3
CHOOSE_QUE = 4
CHOOSE_OTHER_QUE = 5
CHOOSE_QUIT = 6
QUIT = 7


# Define Global var to record stock and question
stocks = []
stocks_reader = Stock('A')
question = ''
repeat = 0


# Define the name and weather of bot
names = ["Stock Helper", "Stock Chatbot"]
weathers = ["cloudy", "sunny", "windy", "rainy"]
name = random.choice(names)
weather = random.choice(weathers)


# Create a trainer that uses this config
trainer = Trainer(config.load("config\config_spacy.yml"))

# Load the training data
training_data = load_data('data\demo-rasa-stock.json')

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Initialize bot
bot = Bot(console_qr=True, cache_path=True)

# Define the policy rules
policy_rules = {
    (INIT, "default"): (INIT, "You'll have to log in first, what's your phone number?", AUTHED),
    (INIT, "phone_number"): (AUTHED, "Perfect, welcome back! Choose the stock you want to know.", None),
    (AUTHED, "default"): (AUTHED, "Would you like to know the information about AAPL or TSLA?", CHOOSE_STOCK),
    (AUTHED, "negative"): (CHOOSE_OTHER_STOCK, "Ok, tell me the stock you want to choose", None),
    (AUTHED, "specify_stock"): (CHOOSE_STOCK, "Perfect, trying to get the infromation", CHOOSE_QUE),
    (CHOOSE_STOCK, "default"): (CHOOSE_STOCK,
                                "Would you like to know about the price or market capitalization?", CHOOSE_QUE),
    (CHOOSE_STOCK, "specify_stock"): (CHOOSE_STOCK,
                                      "Ok, add this to the stock list. What kind of information you want to know?",
                                      CHOOSE_QUE),
    (CHOOSE_STOCK, "negative"): (CHOOSE_OTHER_QUE, "Ok, then I guess you want to know the turnover, right?", None),
    (CHOOSE_STOCK, "specify_question"): (CHOOSE_QUE, "Get that!", None),
    (CHOOSE_OTHER_STOCK, "default"): (CHOOSE_OTHER_STOCK,
                                      "I didn't get that, tell me the stock you want to choose more clearly",
                                      CHOOSE_STOCK),
    (CHOOSE_OTHER_STOCK, "specify_stock"): (CHOOSE_STOCK, "Perfect, trying to get the infromation", CHOOSE_QUE),
    (CHOOSE_QUE, "specify_question"): (CHOOSE_QUIT, "", None),
    (CHOOSE_QUE, "specify_stock"): (CHOOSE_QUIT, "", None),
    (CHOOSE_QUE, "default"): (CHOOSE_QUIT, "", None),
    (CHOOSE_OTHER_QUE, "specify_question"): (CHOOSE_QUIT, "", None),
    (CHOOSE_OTHER_QUE, "default"): (CHOOSE_QUIT, "", None),
    (CHOOSE_QUIT, "default"): (CHOOSE_QUIT, "Would you like to inquire other information or quit?", CHOOSE_QUE),
    (CHOOSE_QUIT, "specify_question"): (CHOOSE_QUE, "OK", None),
    (CHOOSE_QUIT, "quit"): (INIT, "See you next time!", None),
    (CHOOSE_QUIT, "negative"): (AUTHED, "Would you like to know the information about AAPL or TSLA?", CHOOSE_STOCK)
}


# Define rules with patterns and responses
rules = {'I want (.*)': ['What would it mean if you got {0}?',
                         'Why do you want {0}?',
                         "What's stopping you from getting {0}?"],
         'do you remember (.*)': ['Did you think I would forget {0}',
                                  "Why haven't you been able to forget {0}",
                                  'What about {0}',
                                  'Yes .. and?'],
         'do you think (.*)\?': ['if {0}? Absolutely.',
                                 'No chance'],
         'if (.*)': ["Do you really think it's likely that {0}",
                     'Do you wish that {0}',
                     'What do you think about {0}',
                     'Really--if {0}'],
         'how to (.*)\?': ['Why do you want to {0}?',
                           'It is hard to {0}. I do not know.']}


# Define a dict containing responses for statement
statement_responses = {'statement': ['tell me more!',
                                     'why do you think that?',
                                     'how long have you felt this way?',
                                     'I find that extremely interesting',
                                     'can you back that up?',
                                     'oh wow!',
                                     'Sounds boring.',
                                     ':)']}


# Define a dictionary containing a list of responses for each message
responses = {
    "what's your name?": [
        "my name is {0}".format(name),
        "they call me {0}".format(name),
        "I go by {0}".format(name),
        "{0}!".format(name)
        ],
    "default": ["I don't know :(",
                'you tell me!',
                'That is beyond the scope of my knowledge.']}


##################################################################
# Utility functions                                              #
##################################################################

def negated_ents(phrase, ent_vals):
    """
    Find negated entities in the input message
    :param phrase: a string of message
    :param ent_vals: entity values
    :return: dictionary of entities with boolean value
    """
    ents = [e for e in ent_vals if e in phrase]
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    start = 0
    chunks = []
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result


def interpret(message):
    """
    get intent of input message
    :param message: a string of message
    :return: a string of intent
    """
    # Declare global var
    global stocks
    global stocks_reader
    global question
    # Initialize params and neg_params
    params = {}
    neg_params = {}

    # match phone number
    match = re.search('(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                      message)
    if match is not None:
        return 'phone_number'

    # Extract the entities
    interpreted = interpreter.parse(message)
    print(interpreted)
    entities = interpreted["entities"]
    intent = interpreted["intent"]['name']
    ent_vals = [e["value"] for e in entities]
    # Look for negated entities
    negated = negated_ents(message, ent_vals)
    for ent in entities:
        if ent["value"] in negated and negated[ent["value"]]:
            neg_params[ent["entity"]] = str(ent["value"])
        else:
            params[ent["entity"]] = str(ent["value"])

    # Get stock list
    if intent == 'stock_search':
        nlp = spacy.load('en_core_web_md')
        doc = nlp(message)
        possible_pos = ["NOUN", "PROPN"]
        for ent in doc:
            if ent.pos_ in possible_pos \
                    and re.search('(-|[A-Z]){2,8}', ent.text) is not None \
                    and ent.text not in stocks:
                stocks.append(ent.text)
        # deal with some special symbol
        match_special = re.findall('[A-Z]+-[A-Z]?\*?', message)
        if len(match_special) > 0:
            stocks.extend(match_special)
        if re.search(' A$|^A$|A and', message) is not None and 'A' not in stocks:
            stocks.append("A")
        stocks = list(set(stocks))
        # read stocks
        if len(stocks) > 0:
            print(stocks)
            for stock in stocks:
                try:
                    Stock(stock).get_market_cap()
                except iexfinance.utils.exceptions.IEXSymbolError:
                    stocks.remove(stock)
            if len(stocks) == 0:
                return 'unavailable'
            stocks_reader = Stock(stocks)
            return 'specify_stock'

    # Negative intent
    if intent == 'negative':
        return 'negative'

    # ask question intent
    if intent == 'price_search':
        question = 'price'
        return 'specify_question'
    if intent == 'capitalization_search':
        question = 'capitalization'
        return 'specify_question'
    if intent == 'turnover_search':
        question = 'turnover'
        return 'specify_question'

    # greeting
    if intent == 'greet':
        return 'greet'
    return 'default'


##################################################################
# Unrelated chatting functions                                   #
##################################################################
def chitchat_response(message):
    """
    response to chitchat
    :param message: a string of message
    :return: a string of response
    """
    # Call match_rule()
    response, phrase = match_rule(rules, message)
    # Return none is response is "default"
    if response == "default":
        response = basic_respond(message)
    if '{0}' in response:
        # Replace the pronouns of phrase
        phrase = replace_pronouns(phrase)
        # Calculate the response
        response = response.format(phrase)
    return response


def match_rule(rules, message):
    """
    match rules in message and give response
    :param rules: a dictionary of rules
    :param message: a string of massage
    :return: a string of response
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, message)
        if match is not None:
            response = random.choice(responses)
            var = match.group(1) if '{0}' in response else None
            return response, var
    return "default", None


def basic_respond(message):
    """
    basic level response of robot
    :param message: a string of massage
    :return: a string of response
    """
    if message.endswith("?"):
        # Check if the message is in the responses
        if message in responses:
            # Return a random matching response
            response = random.choice(responses[message])
        else:
            # Return a random "default" response
            response = random.choice(responses["default"])
    else:
        # Return a random statement
        response = random.choice(statement_responses["statement"])
    return response


def replace_pronouns(message):
    """
    replace pronouns with different person
    :param message: a string of massage
    :return: a string of replaced massage
    """
    message = message.lower()
    if ' me' in message:
        return re.sub('me', 'you', message)
    if 'i ' in message:
        return re.sub('i ', 'you ', message)
    elif 'my' in message:
        return re.sub('my', 'your', message)
    elif 'your' in message:
        return re.sub('your', 'my', message)
    elif 'you' in message:
        return re.sub('you', 'me', message)
    return message


##################################################################
# Controller functions                                           #
##################################################################
def send_message(state, pending, message):
    """
    Transition of Finite State Machine
    :param state: a string of state
    :param pending: a string of pending state
    :param message: a string of message
    :return: new state
    """
    global stocks
    global stocks_reader
    global question
    global repeat
    res = ''
    intent = interpret(message)
    print(state, intent, pending)
    if intent == "unavailable":
        res += "{}".format("There is no available stock symbol I can find, please capitalize the symbol\n")
    # Deal with chitchat
    if ((state, intent) not in policy_rules) or intent == 'default':
        if intent == 'greet':
            res += "{}\n".format('Hi!')
            return state, pending, res
        response = chitchat_response(message)
        if response is not None and intent != "unavailable":
            res += "{}\n".format(response)
        pending = None
        new_state, response, pending_state = policy_rules[(state, "default")]
        # Only repeat the requirement first time in the same state
        if new_state == state:
            repeat += 1
        else:
            repeat = 0
        if repeat < 2 or (state == CHOOSE_STOCK and intent == 'specify_stock'):
            res += "{}\n".format(response)
        if pending_state is not None:
            pending = (pending_state, "default")
        return new_state, pending, res

    # Calculate the new_state, response, and pending_state
    new_state, response, pending_state = policy_rules[(state, intent)]
    res += "{}\n".format(response)
    print(21, new_state, response, pending_state)

    # Pending state transitions
    if pending is not None:
        if (pending[0] == AUTHED and intent == 'phone_number') \
                or (pending[0] == CHOOSE_STOCK and intent == 'specify_stock') \
                or (pending[0] == CHOOSE_QUE and intent == 'specify_question'):
            print(22, pending[0], new_state, question)
            # print question answer
            if question == "price":
                if len(stocks) > 1:
                    for i in range(len(stocks)):
                        res += "The price of {} is {}\n".format(
                            stocks_reader.get_company_name()[stocks[i]],
                            stocks_reader.get_price()[stocks[i]])
                else:
                    res += "The price of {} is {}\n".format(
                        stocks_reader.get_company_name(),
                        stocks_reader.get_price())
            if question == "capitalization":
                if len(stocks) > 1:
                    for i in range(len(stocks)):
                        res += "The market capitalization  of {} is {}\n".format(
                            stocks_reader.get_company_name()[stocks[i]],
                            stocks_reader.get_market_cap()[stocks[i]])
                else:
                    res += "The market capitalization  of {} is {}\n".format(
                        stocks_reader.get_company_name(),
                        stocks_reader.get_market_cap())
            if question == "turnover":
                if len(stocks) > 1:
                    for i in range(len(stocks)):
                        res += "The turnover of {} is {}\n".format(
                            stocks_reader.get_company_name()[stocks[i]],
                            stocks_reader.get_volume()[stocks[i]])
                else:
                    res += "The turnover of {} is {}\n".format(
                        stocks_reader.get_company_name(),
                        stocks_reader.get_volume())
            new_state, response, pending_state = policy_rules[pending]
            if response != "":
                res += "{}\n".format(response)
            else:
                new_state, response, pending_state = policy_rules[(new_state, "default")]
                res += "{}\n".format(response)
            intent = "default"

    if pending_state is not None:
        pending = (pending_state, intent)

    print(23, new_state, pending, res)
    return new_state, pending, res


# Define send_messages()
def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        # print(state, pending, msg)
        state, pending = send_message(state, pending, msg)
        # print(state, pending, msg)


@bot.register(bot.file_helper, except_self=False)
def reply_self(msg):
    global state
    global pending
    print(0)
    if msg.type != 'Text':
        print(1)
        response = "[奸笑][奸笑]I don't understand images"
    elif msg.text == 'quit':
        response = " Bye!"
        state = INIT
        pending = None
    else:
        print(2)
        print(msg.text, state, pending)
        state, pending, response = send_message(state, pending, msg.text)
        print(3, state, pending, response)
    bot.file_helper.send('BOT:' + response)


##################################################################
# Main                                                           #
##################################################################
if __name__ == '__main__':
    bot.file_helper.send("BOT: Hello, I'm stock helper, you can use phone number to log in.")
    # Initialize state and pending
    state = INIT
    pending = None
    embed()
    # message = input(username + ": ")
    # while message != 'quit':
    #     state, pending = send_message(state, pending, message)
    #     message = input(username + ": ")


##################################################################
# FOR TEST                                                       #
##################################################################
# send_messages([
#     "what's your name?",
#     "55s-12345",
#     "my phone number is 555-12345",
#     "No",
#     "i'm looking for A",
#     "AAPL",
#     "QTUMUSDT",
#     "price",
#     "No, show me the turnover",
#     "market cap"
# ])
#
# aapl = Stock(["aaPL", "A"])
# try:
#     print(aapl.get_market_cap())
# except iexfinance.utils.exceptions.IEXSymbolError:
#     print("Not found that stock")
# stockss = iexfinance.get_available_symbols()
# for adasd in stockss:
#     if '-' in adasd['symbol']:
#         print(adasd['symbol'])
# ss = []
# for s in stockss:
#     ss.append(s['symbol'])
# print(ss)
