import os
import time
from pathlib import Path
import openai

def main():
    # Load your OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Please set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'.")

    # Initialize the OpenAI client
    openai.api_key = api_key
    client = openai

    # Path to your training data file
    training_file_path = 'training_data_no_comments.jsonl'

    # Check if the training data file exists
    if not os.path.exists(training_file_path):
        raise FileNotFoundError(f"Training data file '{training_file_path}' not found.")

    # Upload the training file
    print("Uploading training file...")
    response = client.files.create(
        file=Path(training_file_path),
        purpose='fine-tune'
    )
    training_file_id = response.id
    print(f"Uploaded training file. File ID: {training_file_id}")

    # Create the fine-tuning job
    print("Creating fine-tuning job...")
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model='gpt-4o-2024-08-06',  # Or 'gpt-4' if you have access
        suffix='sequence_predictor_v0.02',  # Optional: custom suffix
        hyperparameters={
            "n_epochs": 1
        }
    )
    fine_tune_job_id = fine_tune_response.id
    print(f"Fine-tuning job created. Job ID: {fine_tune_job_id}")

    # Monitor the fine-tuning job
    print("Monitoring fine-tuning job. This may take some time...")
    while True:
        # Retrieve job status
        job_status = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
        status = job_status.status
        print(f"Job status: {status}")

        if status == 'succeeded':
            fine_tuned_model = job_status['fine_tuned_model']
            print("Fine-tuning job completed successfully.")
            print(f"Fine-tuned model name: {fine_tuned_model}")
            break
        elif status == 'failed':
            print("Fine-tuning job failed.")
            # Optionally, print error information
            print(f"Failure reason: {job_status.get('failure_reason', 'Unknown')}")
            break
        else:
            # Wait before checking again
            time.sleep(60)  # Check every 60 seconds

    # Save the fine-tuned model name for future use
    if status == 'succeeded':
        with open('fine_tuned_model_name.txt', 'w') as f:
            f.write(fine_tuned_model)
        print("Fine-tuned model name saved to 'fine_tuned_model_name.txt'.")

if __name__ == '__main__':
    main()
