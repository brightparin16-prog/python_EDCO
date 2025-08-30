import random

def guess_the_number():
    """
    This function implements the 'Guess the Number' game.
    The player tries to guess a randomly generated number within a certain range.
    """
    print("Welcome to 'Guess the Number'!")
    print("I'm thinking of a number between 1 and 100.")
    print("Can you guess it?")

    # Generate a random number between 1 and 100 (inclusive)
    secret_number = random.randint(1, 100)
    attempts = 0
    guess = 0

    while guess != secret_number:
        try:
            # Get player's guess
            guess = int(input("Enter your guess: "))
            attempts += 1

            # Provide hints
            if guess < secret_number:
                print("Too low! Try again.")
            elif guess > secret_number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number {secret_number} in {attempts} attempts.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("Thanks for playing!")

# Run the game
if __name__ == "__main__":
    guess_the_number()
