from src.chatbot import Chatbot

def do_chat(text):
    c = Chatbot()

def main():
    print("TRACKBOT COMMAND LINE INTERFACE TEST\n")
    
    # Chat loop
    c = Chatbot()
    while True:
        print(c.get_response(input()))

if __name__ == "__main__":
    main()
