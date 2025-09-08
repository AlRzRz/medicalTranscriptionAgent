import time

# GPT-like fluid text streaming into the terminal
def streamText(text):

    for char in text:
        time.sleep(0.015)
        print(char, end='', flush=True)

    print()


def main():
    
    streamText("How can I help you today?")


if __name__ == '__main__':
    main()