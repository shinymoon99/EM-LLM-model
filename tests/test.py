import pandas as pd
from datasets import load_from_disk, Dataset
import os
import pyarrow as pa

def read_hotpotqa_data(data_dir="benchmark/data/longbench/hotpotqa", num_examples=5):
    """
    Read and display examples from the HotpotQA dataset in the LongBench collection.
    
    Args:
        data_dir: Path to the directory containing the dataset
        num_examples: Number of examples to display
    
    Returns:
        The loaded dataset
    """
    try:
        # Method 1: Using the datasets library to load from disk
        # The data is in Arrow format in the main directory
        dataset = load_from_disk(data_dir)
        
        # If load_from_disk doesn't work, try loading the Arrow file directly
        # if dataset is None:
        #     arrow_path = os.path.join(data_dir, "data-00000-of-00001.arrow")
        #     arrow_table = pa.ipc.open_file(arrow_path).read_all()
        #     dataset = Dataset(arrow_table=arrow_table)
        
        # Print some examples
        print(f"Successfully loaded HotpotQA dataset with {len(dataset)} examples")
        print("\n=== Sample Examples ===")
        dataset = list(dataset)
        for i, example in enumerate(dataset[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Question: {example['input']}")
            print(f"Context length: {example['length']} characters")
            print(f"Answer: {example['answers'][0] if example['answers'] else 'No answer provided'}")
            # Printing just the beginning of the context to avoid overwhelming output
            context_preview = example['context'][:200] + "..." if len(example['context']) > 200 else example['context']
            print(f"Context (preview): {context_preview}")
            print("-" * 50)
            
        return dataset
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the path to the dataset is correct")
        print("2. Ensure you have the 'datasets' and 'pyarrow' libraries installed")
        print("3. Check if the arrow file exists in the specified directory")
        print(f"4. Current directory structure: {os.listdir(data_dir) if os.path.exists(data_dir) else 'Directory not found'}")
        
        # Try alternative loading method as a fallback
        try:
            print("\nAttempting alternative loading method...")
            from datasets import load_dataset
            dataset = load_dataset("THUDM/LongBench", "hotpotqa", split="test")
            print(f"Successfully loaded dataset from Hugging Face Hub with {len(dataset)} examples")
            return dataset
        except Exception as fallback_error:
            print(f"Fallback loading also failed: {fallback_error}")
        
        return None

if __name__ == "__main__":
    dataset = read_hotpotqa_data()