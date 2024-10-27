import os
import openai  # Use 'openai' instead of 'from openai import OpenAI'

def load_fine_tuned_model():
    """
    Loads the fine-tuned model name from a file.
    """
    # v0.00 model; old comments
    # v0.01 model; new format of %N and comments, 1k examples
    # v0.02 model; same format of v0.01, but without comments and 10k examples, 1 epoch
    return 'ft:gpt-4o-2024-08-06:personal:sequence-predictor-v0-02:AMmi6oee'

def generate_prediction(input_sequence, fine_tuned_model=None):
    """
    Generates a prediction using the fine-tuned model for a given input sequence.

    Parameters:
    - input_sequence (str): The initial terms of the sequence as a string.
    - fine_tuned_model (str): The name of the fine-tuned model (optional).

    Returns:
    - assistant_reply (str): The assistant's response containing comments and the next terms.
    """
    if fine_tuned_model is None:
        fine_tuned_model = load_fine_tuned_model()

    # Initialize the OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    if openai.api_key is None:
        raise ValueError("Please set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'.")

    # Prepare the input message
    sequence_input = f"[BEGINNING SEQ]\nSequence: {input_sequence}\n[END SEQ]"

    # Generate a completion
    chat_completion = openai.chat.completions.create(
        model=fine_tuned_model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that, given the beginning of an integer sequence, predicts the description and the next terms."
            },
            {
                "role": "user",
                "content": sequence_input
            }
        ],
        max_tokens=100,  # Adjust as needed
        temperature=0  # Set temperature to 0 for deterministic output
    )

    # Extract the assistant's reply
    assistant_reply = chat_completion.choices[0].message.content
    return assistant_reply

def test_model_on_sequences(input_sequences, fine_tuned_model=None):
    """
    Tests the fine-tuned model on a list of input sequences and checks the responses.

    Parameters:
    - input_sequences (list): A list of dictionaries, each with 'name' and 'full_sequence' keys.
    - fine_tuned_model (str): The name of the fine-tuned model (optional).
    """
    # Load the fine-tuned model name if not provided
    if fine_tuned_model is None:
        fine_tuned_model = load_fine_tuned_model()

    # Iterate over test cases
    for test in input_sequences:
        print(f"\nTesting sequence: {test['name']}")
        full_sequence = test['full_sequence']
        n = len(full_sequence)
        if n < 2:
            print("Sequence is too short to split into two halves.")
            continue

        split_index = n // 2
        input_sequence = ', '.join(map(str, full_sequence[:split_index]))  # First half
        output_sequence = ', '.join(map(str, full_sequence[split_index:]))  # First half
        assistant_reply = generate_prediction(input_sequence, fine_tuned_model)
        print(f"[BEGINNING SEQ]\nSequence: {input_sequence}\n[END SEQ]")
        print("Assistant's reply:")
        print(assistant_reply)
        print('\nCorrect Sequence: ', output_sequence)
        print('-' * 20)

    print("\nTesting completed.")

# Example usage
if __name__ == '__main__':
    # Define test cases: each is a dictionary with the full sequence
    test_cases = [
        {
            "name": "Powers of 2",
            "full_sequence": [
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288
            ],
        },
        {
            "name": "Prime Numbers",
            "full_sequence": [
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71
            ],
        },
        {
            "name": "Fibonacci Sequence",
            "full_sequence": [
                0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
                55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181
            ],
        },
        {
            "name": "Squares",
            "full_sequence": [
                1, 4, 9, 16, 25, 36, 49, 64, 81, 100,
                121, 144, 169, 196, 225, 256, 289, 324, 361, 400
            ],
        },
        {
            "name": "Triangular Numbers",
            "full_sequence": [
                1, 3, 6, 10, 15, 21, 28, 36, 45, 55,
                66, 78, 91, 105, 120, 136, 153, 171, 190, 210
            ],
        },
        {
            "name": "Cubes",
            "full_sequence": [
                1, 8, 27, 64, 125, 216, 343, 512, 729, 1000,
                1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000
            ],
        },
        {
            "name": "Factorials",
            "full_sequence": [
                1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
                3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000
            ],
        },
        {
            "name": "Powers of 3",
            "full_sequence": [
                1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683,
                59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163, 387420489, 1162261467
            ],
        },
        {
            "name": "Pentagonal Numbers",
            "full_sequence": [
                1, 5, 12, 22, 35, 51, 70, 92, 117, 145,
                176, 210, 247, 287, 330, 376, 425, 477, 532, 590
            ],
        },
        {
            "name": "Hexagonal Numbers",
            "full_sequence": [
                1, 6, 15, 28, 45, 66, 91, 120, 153, 190,
                231, 276, 325, 378, 435, 496, 561, 630, 703, 780
            ],
        },
        {
            "name": "Catalan Numbers",
            "full_sequence": [
                1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862,
                16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190
            ],
        },
        {
            "name": "Harmonic Numbers (First 20 Approximations)",
            "full_sequence": [
                1, 1.5, 1.8333, 2.0833, 2.2833, 2.45, 2.5929, 2.7179, 2.8289, 2.9289,
                3.0199, 3.1032, 3.1801, 3.2516, 3.3182, 3.3807, 3.4395, 3.4951, 3.5477, 3.5977
            ],
        },
        {
            "name": "Lucas Numbers",
            "full_sequence": [
                2, 1, 3, 4, 7, 11, 18, 29, 47, 76,
                123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349
            ],
        },
        {
            "name": "Tetrahedral Numbers",
            "full_sequence": [
                1, 4, 10, 20, 35, 56, 84, 120, 165, 220,
                286, 364, 455, 560, 680, 816, 969, 1140, 1330, 1540
            ],
        },
        {
            "name": "Octagonal Numbers",
            "full_sequence": [
                1, 8, 21, 40, 65, 96, 133, 176, 225, 280,
                341, 408, 481, 560, 645, 736, 833, 936, 1045, 1160
            ],
        },
        {
            "name": "Niven Numbers (Harshad Numbers under 100)",
            "full_sequence": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                12, 18, 20, 21, 24, 27, 30, 36, 40, 42
            ],
        },
    ]
    test_model_on_sequences(test_cases)
