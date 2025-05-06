import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1"
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("❌ Please set the OPENROUTER_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

def get_last_id_and_count(filename: str) -> tuple[int, int]:
    """Reads a JSONL file, counts valid lines, and returns the last ID number and count."""
    last_id_num = 0
    line_count = 0
    last_valid_line = None
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "id" in data:
                                last_valid_line = data # Keep track of the last valid line's data
                                line_count += 1
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping malformed JSON line during initial read: {line}", file=sys.stderr)
            
            if last_valid_line and isinstance(last_valid_line.get("id"), str) and last_valid_line["id"].startswith("JD_"):
                 try:
                     last_id_num = int(last_valid_line["id"].split("_")[-1])
                 except (ValueError, IndexError):
                     print(f"⚠️ Could not parse ID number from last valid line: {last_valid_line['id']}", file=sys.stderr)
                     last_id_num = 0 # Or potentially raise an error if this is critical

        except Exception as e:
            print(f"❌ Error reading existing file {filename}: {e}", file=sys.stderr)
            return 0, 0 # Return defaults if file read fails critically
            
    return last_id_num, line_count

def call_openrouter(total_count: int, output_filename: str):
    """
    Calls OpenRouter API, streams response, parses JSONL, saves incrementally,
    and resumes if the output file already exists.
    """
    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=API_KEY,
    )

    last_id_num, lines_already_present = get_last_id_and_count(output_filename)
    
    start_id = last_id_num + 1
    remaining_count = total_count - lines_already_present

    if remaining_count <= 0:
        print(f"✅ Target count of {total_count} already met or exceeded in {output_filename}. Nothing to do.")
        return

    print(f"ℹ️ Found {lines_already_present} existing jobs. Attempting to generate {remaining_count} more (IDs starting from {start_id}).")

    lines_written_this_run = 0
    line_buffer = ""
    request_made = False # Flag to track if API call was initiated

    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        
        # Open in append mode ('a+') which creates if not exists
        with open(output_filename, "a+", encoding="utf-8") as f, \
             tqdm.tqdm(total=total_count, initial=lines_already_present, desc="Generating Jobs", unit="job") as pbar:
            
            # Seek to the end of the file before appending if it existed
            if lines_already_present > 0:
                 f.seek(0, os.SEEK_END)
                 # Add a newline if the file doesn't end with one, to prevent merging JSON objects
                 if f.tell() > 0:
                     f.seek(f.tell() - 1, os.SEEK_SET)
                     if f.read(1) != '\n':
                         f.write('\n')

            stream = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free", # Or your preferred model
                messages=[
                    {
                        "role": "system",
                        # Pass remaining_count and start_id to the prompt function
                        "content": get_prompt(remaining_count, start_id=start_id), 
                    },
                    {
                        "role": "user",
                        "content": "EXECUTE",
                    },
                ],
                stream=True,
            )
            request_made = True # Mark that the API call started

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    line_buffer += chunk.choices[0].delta.content

                    # Process complete lines from the buffer
                    while '\n' in line_buffer:
                        line, line_buffer = line_buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                
                                f.write(line + "\n")
                                f.flush() # Ensure it's written to disk immediately
                                lines_written_this_run += 1
                                pbar.update(1)
                                if lines_written_this_run >= remaining_count: # Stop if we have generated the required number for this run
                                    break 
                            except json.JSONDecodeError:
                                print(f"\n⚠️ Skipping invalid JSON line: {line}", file=sys.stderr)
                if lines_written_this_run >= remaining_count:
                     break # Exit outer loop too

            # Process any remaining content in the buffer after the stream ends
            if line_buffer.strip() and lines_written_this_run < remaining_count:
                 line = line_buffer.strip()
                 try:
                     json.loads(line) # Validate
                     f.write(line + "\n")
                     lines_written_this_run += 1
                     pbar.update(1)
                 except json.JSONDecodeError:
                     print(f"\n⚠️ Skipping invalid JSON line at end: {line}", file=sys.stderr)
            
            # Add a small delay if needed
            # time.sleep(0.5) 

    except Exception as e:
        print(f"\n❌ An error occurred during generation or writing: {e}", file=sys.stderr)
    finally:
        # Recalculate final count from file for accuracy
        _, final_lines_in_file = get_last_id_and_count(output_filename)
        pbar.n = final_lines_in_file # Ensure pbar reflects actual file content
        pbar.refresh()

        if not request_made and remaining_count > 0:
             print("\n⚠️ API request was not made. No new jobs generated.")
        elif final_lines_in_file < total_count:
            print(f"\n⚠️ Expected {total_count} total lines but file only contains {final_lines_in_file}.", file=sys.stderr)
        elif final_lines_in_file > total_count:
             # This case might be less common now with stricter checks, but good to keep
             print(f"\n⚠️ Wrote more lines than expected. File contains {final_lines_in_file} lines, target was {total_count}.", file=sys.stderr)
        
        print(f"✅ Finished. {output_filename} now contains {final_lines_in_file} jobs.")

# Modify get_prompt to accept start_id
def get_prompt(count: int, start_id: int = 1) -> str:
    end_id = start_id + count - 1
    return f"""
You are a data-generation assistant. Produce exactly {count} distinct, well-written software-developer
job descriptions, each 80–150 characters long, containing 1–4 explicit skill mentions drawn from technologies
like Python, JavaScript, React, Angular, AWS, Docker, Kubernetes, SQL, NoSQL, TypeScript, Django, Flask, etc.

OUTPUT FORMAT (JSONL, one object per line, no surrounding array):
{{ 
  "id": "JD_XXX",             
  "text": "<full job description>",
  "entities": [
    {{ "label": "SKILL", "text": "<exact span>" }}
    …
  ]
}}

CONSTRAINTS:
1. Do NOT include `start` or `end` offsets—only "label" and "text".
2. Each "text" sentence must slice cleanly; entities should match exact substrings.
3. IDs must increment exactly from JD_{start_id:03} to JD_{end_id:03}.
4. Output must be valid JSONL (one JSON object per line).

After these instructions, output only the {count} JSONL lines—no additional commentary.
Once the user parses the command "EXECUTE", you can start generating the job descriptions.
""".strip()

def save_jsonl(raw: str, filename: str, expected_count: int):
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != expected_count:
        print(f"⚠️ Expected {expected_count} lines but got {len(lines)}. Saving anyway.", file=sys.stderr)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


if __name__ == "__main__":
    COUNT = 500
    OUTPUT_FILE = "jobs-new.jsonl"

    print(f"Generating {COUNT} job descriptions...")
    call_openrouter(COUNT, OUTPUT_FILE)
    print(f"Job descriptions saved to {OUTPUT_FILE}.")
