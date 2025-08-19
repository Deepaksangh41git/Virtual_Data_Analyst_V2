import os
import shutil
import logging
import uuid
import json
import re
import subprocess
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Comment
import google.generativeai as genai

# Configure your Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
# genai.configure(api_key="your-api-key-here")
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/data_analyst_agent"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

@app.post("/api/")
async def handle_upload(request: Request):
    try:
        # Get form data from request
        form = await request.form()
        
        # Create a unique request folder inside /tmp
        request_id = str(uuid.uuid4())
        request_folder = os.path.join(UPLOAD_DIR, request_id)
        os.makedirs(request_folder, exist_ok=True)
        
        saved_files = {}
        has_question_file = False
        questions_file_path = None
        
        # Process each item in the form
        processed_files = set()  # Track processed files to avoid duplicates
        
        for key, val in form.items():
            print(f"Form key: {key}, Val type: {type(val)}")  # Debug log
            
            # Check if it's a single file upload
            if hasattr(val, "filename") and val.filename:
                filename = val.filename
                
                # Skip if we've already processed this file
                if filename in processed_files:
                    print(f"Skipping duplicate file: {filename}")
                    continue
                    
                file_path = os.path.join(request_folder, filename)
                
                # Read file content and save to disk
                content = await val.read()
                print(f"File {filename} content length: {len(content)}")  # Debug file size
                
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
                
                logging.info(f"Saved file: {filename} -> {file_path}")
                saved_files[filename] = file_path
                processed_files.add(filename)  # Mark as processed
                
                # Check if this is the questions file
                if filename.lower() in ["questions.txt", "question.txt"]:
                    has_question_file = True
                    questions_file_path = file_path
                    print(f"Found questions.txt at: {file_path}")
        
        # Also check if 'files' key contains multiple files (only if we haven't processed them individually)
        if 'files' in form:
            files_val = form.getlist('files')  # Get all files with 'files' key
            print(f"Found {len(files_val)} files in 'files' key")
            
            for file_item in files_val:
                if hasattr(file_item, "filename") and file_item.filename:
                    filename = file_item.filename
                    
                    # Skip if we've already processed this file
                    if filename in processed_files:
                        print(f"Skipping duplicate file: {filename}")
                        continue
                        
                    file_path = os.path.join(request_folder, filename)
                    
                    content = await file_item.read()
                    print(f"File {filename} content length: {len(content)}")  # Debug file size
                    
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)
                    
                    logging.info(f"Saved file: {filename} -> {file_path}")
                    saved_files[filename] = file_path
                    processed_files.add(filename)  # Mark as processed
                    
                    if filename.lower() in ["questions.txt", "question.txt"]:
                        has_question_file = True
                        questions_file_path = file_path
                        print(f"Found questions.txt at: {file_path}")
        
        print(f"All saved files: {list(saved_files.keys())}")
        print(f"Has questions file: {has_question_file}")
        
        # Validate that questions.txt was uploaded
        if not has_question_file:
            raise HTTPException(status_code=400, detail="Missing required file: questions.txt")
        
        # Read content of questions.txt
        with open(questions_file_path, "r") as f:
            question_content = f.read()
        
        print("Questions file read successfully")
        
        # Analyze the question content
        detect_prompt = f"""
        Analyze this question and return a JSON object with these exact keys:

        - "scraping": "yes" or "no"  
        - "urls": list of URLs found in the question (empty list if none)
        - "data_type": "table" or "list" or "text"
        - "target_elements": list of HTML elements to target
        - "steps": list of steps to solve the question

        Question: {question_content}
        
        Return only valid JSON, no other text or formatting.
        """

        try:
            detection_result = model.generate_content(detect_prompt)
            detection_json = extract_json(detection_result.text.strip())
            print("Detection result:", detection_json)
        except Exception as e:
            print(f"Error in detection phase: {e}")
            return JSONResponse({
                "error": f"Failed to analyze question: {str(e)}",
                "question": question_content.strip()
            })

        # Handle scraping flow
        if detection_json["scraping"].lower() == "yes" and detection_json.get("urls"):
            # Check if URLs were detected
            if not detection_json["urls"]:
                return JSONResponse({
                    "error": "Scraping requested but no URLs found in the question",
                    "question": question_content.strip(),
                    "detection_result": detection_json
                })
            
            url = detection_json["urls"][0]
            
            # Enhanced scraping with better HTML processing
            scraped_data = await enhanced_scrape_and_analyze_with_verification(
                url=url,
                question=question_content,
                detection_info=detection_json,
                work_folder=request_folder
            )
            
            return JSONResponse(scraped_data)
        else:
            # Handle file-based analysis (CSV, PDF, images, etc.)
            print("in the else part bro")
            file_analysis_data = await enhanced_file_analysis_with_verification(
                question=question_content,
                detection_info=detection_json,
                saved_files=saved_files,
                work_folder=request_folder
            )
            
            return JSONResponse(file_analysis_data)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
        
async def enhanced_scrape_and_analyze_with_verification(url: str, question: str, detection_info: dict, work_folder: str):
    """Enhanced scraping with result verification - keeps trying until same result twice"""
    print("in the 2nd step ")
    # Step 1: Get HTML content (same as before)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
            html_content = await page.content()
            await browser.close()
    except Exception as e:
        return {"error": f"Failed to fetch page: {str(e)}"}

    # Step 2: Process HTML
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    html_file_path = os.path.join(work_folder, "scraped_page.html")
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    data_preview = extract_data_preview(soup, detection_info.get("data_type", "table"))
    
    # Step 3: Generate and verify results
    return await generate_and_verify_results(
        question=question,
        detection_info=detection_info,
        html_file_path=html_file_path,
        data_preview=data_preview,
        work_folder=work_folder,
        max_attempts=6,
        verification_rounds=5  # Maximum verification rounds
    )


async def generate_and_verify_results(question: str, detection_info: dict, html_file_path: str, 
                                    data_preview: dict, work_folder: str, max_attempts: int = 6, 
                                    verification_rounds: int = 5, custom_tolerance: dict = None):
    """
    Simple flow: Code Generation LLM ↔ Code Execution until no runtime errors, then return result
    """

    base_prompt = f"""
You are a Python web scraping expert. Write code to extract data and answer this question:

QUESTION: {question}

DATA LOCATION: {html_file_path}
TARGET DATA TYPE: {detection_info.get("data_type", "unknown")}
TARGET ELEMENTS: {detection_info.get("target_elements", [])}

SAMPLE DATA STRUCTURE (from the page):
{data_preview}

STEPS TO IMPLEMENT:
{json.dumps(detection_info.get("steps", []), indent=2)}

REQUIREMENTS:
1. Load HTML from the file path: {html_file_path}
2. Use BeautifulSoup to parse HTML
3. Extract data based on the sample structure shown above
4. Perform all required calculations/analysis
5. Return results in the exact format requested in the question
6. Handle missing data gracefully (use empty string "" or 0, not None/null)
7. For visualizations, save as PNG and encode as base64 data URI

CRITICAL RULES:
- Import all necessary libraries (pandas, matplotlib, numpy, base64, io, etc.)
- Never hardcode answers - compute everything from scraped data  
- Print only the final JSON result
- Use the actual HTML file content, not placeholder data
- For tables: look for <table>, <tr>, <td> elements
- For lists: look for <ul>, <ol>, <li> elements
- Clean and convert data types properly (strings to numbers where needed)
- Be consistent in your data extraction and calculations
- For numerical results, use sufficient precision but don't over-specify
- Don't give null or empty answers better to give the error dont use exceptional handling
- I want final output as json and dont give array or list as output give only json like this: 
    {{"question 1" : "answer 1 ","question 2 " : "answer 2 "}} 
Write the complete Python code:
"""

    print(f"\n🔍 Starting code generation and execution process...")
    
    for round_num in range(verification_rounds):
        print(f"\n--- Generation Round {round_num + 1} ---")
        
        # Code Generation + Execution Loop (until no runtime errors)
        execution_result = await code_generation_execution_loop(
            prompt=base_prompt,
            work_folder=work_folder,
            html_file_path=html_file_path,
            question=question,
            max_attempts=max_attempts,
            round_num=round_num + 1
        )
        
        if execution_result.get("error"):
            print(f"❌ Round {round_num + 1} failed: {execution_result.get('error')}")
            continue
        
        # Success! Parse and return the result
        result_answer = execution_result.get("answer", "")
        
        print(f"🎉 SUCCESS! Round {round_num + 1} - parsing and returning result")
        
        # Try to parse as JSON and return the parsed object directly
        try:
            parsed_result = json.loads(result_answer)
            # Return the parsed JSON object directly, not wrapped in "answer"
            return parsed_result
        except json.JSONDecodeError:
            # If not valid JSON, still try to extract meaningful data
            print(f"⚠️ Result is not valid JSON, returning as string: {result_answer[:100]}...")
            return {"result": result_answer}
    
    # If all rounds fail
    return {
        "question": question,
        "error": f"All {verification_rounds} rounds failed to generate working code",
        "status": "FAILED",
        "total_rounds": verification_rounds
    }


async def code_generation_execution_loop(prompt: str, work_folder: str, html_file_path: str, 
                                       question: str, max_attempts: int, round_num: int):
    """
    Loop between Code Generation LLM and Execution until no runtime errors
    """
    attempt = 0
    last_error = None
    current_prompt = prompt
    
    while attempt < max_attempts:
        attempt += 1
        print(f"   Code Generation Attempt {attempt} of round {round_num}")
        
        # Generate code from LLM
        try:
            code_response = model.generate_content(current_prompt)
            generated_code = extract_code_blocks(code_response.text)
        except Exception as e:
            last_error = f"LLM generation error: {str(e)}"
            continue
        
        # Execute the code
        code_file = save_code_to_file(generated_code, work_folder, f"round_{round_num}_attempt_{attempt}")
        
        try:
            result = subprocess.run(
                ["python3", code_file],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_folder
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if result.returncode == 0 and stdout:
                print(f"   ✅ Execution successful on attempt {attempt}")
                return {
                    "answer": stdout,
                    "code": generated_code,
                    "round": round_num,
                    "attempt": attempt
                }
            else:
                last_error = stderr or "No output generated"
                print(f"   ❌ Execution Error: {last_error[:100]}...")
                
                # Prepare feedback prompt for next attempt
                current_prompt = f"""
The previous code failed with this error. Fix the issue and generate working Python code:

ERROR: {last_error}

ORIGINAL QUESTION: {question}
WORK DIRECTORY: {work_folder}
HTML FILE: {html_file_path}

FAILED CODE:
{generated_code}

REQUIREMENTS:
- Fix the error above
- Use correct file paths and libraries
- Handle edge cases better
- Extract real data from HTML file
- Return results in exact format requested with question 
- Print only the final JSON result

Generate the corrected Python code:
"""
                
        except subprocess.TimeoutExpired:
            last_error = "Code execution timeout (60 seconds)"
            print(f"   ⏰ Timeout on attempt {attempt}")
        except Exception as e:
            last_error = f"Execution error: {str(e)}"
            print(f"   ❌ Exception on attempt {attempt}: {str(e)}")
    
    return {"error": f"Code generation failed after {max_attempts} attempts. Last error: {last_error}"}


def extract_code_blocks(text: str) -> str:
    """Extract Python code from LLM response"""
    # Look for fenced code blocks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no fenced blocks, look for code-like patterns
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Start of code indicators
        if any(line.strip().startswith(x) for x in ['import ', 'from ', 'def ', 'class ', 'if __name__']):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # Fallback - return entire text
    return text.strip()


def save_code_to_file(code: str, work_folder: str, filename_prefix: str) -> str:
    """Save generated code to a file and return the file path"""
    import os
    filename = f"{filename_prefix}.py"
    filepath = os.path.join(work_folder, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return filepath
def is_result_valid(result) -> bool:
    """
    Check if result is valid - does not contain empty, null, zero, nan, or 'not available' values
    
    Args:
        result: The result to validate (can be string, dict, list, etc.)
        
    Returns:
        bool: True if result is valid, False if it contains invalid data
    """
    if not result:
        return False
    
    # Convert to string for analysis
    result_str = str(result).strip().lower()
    
    # Quick check for completely empty result
    if not result_str:
        return False
    
    # Define invalid patterns to look for
    invalid_patterns = [
        'null', 'none', 'nan', 'na', 'n/a', 'not available', 'not found',
        'empty', 'no data', 'no result', 'undefined', 'missing',
        'error', 'failed', 'exception', 'traceback', 'timeout'
    ]
    
    # Check if result is just invalid patterns
    for pattern in invalid_patterns:
        if result_str == pattern:
            return False
    
    # Try to parse as JSON and validate structure
    try:
        parsed = json.loads(str(result))
        return validate_json_structure(parsed)
    except json.JSONDecodeError:
        # Not JSON, validate as text
        return validate_text_result(result_str, invalid_patterns)


def validate_json_structure(data) -> bool:
    """Validate JSON data structure for meaningful content"""
    
    if isinstance(data, dict):
        if not data:  # Empty dict
            return False
        
        # Check if all values are invalid
        valid_values = 0
        total_values = 0
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # Recursively check nested structures
                if validate_json_structure(value):
                    valid_values += 1
                total_values += 1
            else:
                total_values += 1
                if is_value_meaningful(value):
                    valid_values += 1
        
        # Need at least some valid values
        return valid_values > 0 and (valid_values / total_values) > 0.3
    
    elif isinstance(data, list):
        if not data:  # Empty list
            return False
        
        # Check if list has meaningful items
        meaningful_items = 0
        for item in data:
            if isinstance(item, (dict, list)):
                if validate_json_structure(item):
                    meaningful_items += 1
            else:
                if is_value_meaningful(item):
                    meaningful_items += 1
        
        return meaningful_items > 0
    
    else:
        # Single value
        return is_value_meaningful(data)


def validate_text_result(text: str, invalid_patterns: list) -> bool:
    """Validate text result for meaningful content"""
    
    # Check if text contains mostly invalid patterns
    invalid_count = sum(1 for pattern in invalid_patterns if pattern in text)
    
    # If more than 2 invalid patterns found, likely invalid
    if invalid_count > 2:
        return False
    
    # Look for meaningful content indicators
    # Numbers (excluding just zeros)
    numbers = re.findall(r'-?\d+\.?\d+', text)
    meaningful_numbers = [n for n in numbers if float(n) != 0.0]
    
    # Meaningful words (excluding common invalid patterns)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    meaningful_words = [w for w in words if w.lower() not in invalid_patterns and len(w) > 2]
    
    # Check for data-like content (tables, lists, structured data)
    has_structure = any(indicator in text for indicator in ['[', '{', '|', ':', '-', 'data:', 'result:', 'answer:'])
    
    # Result is valid if it has meaningful numbers, words, or structure
    has_meaningful_content = (
        len(meaningful_numbers) > 0 or 
        len(meaningful_words) > 3 or
        (has_structure and len(meaningful_words) > 1)
    )
    
    return has_meaningful_content


def is_value_meaningful(value) -> bool:
    """Check if a single value is meaningful"""
    if value is None:
        return False
    
    if isinstance(value, str):
        value_clean = value.strip().lower()
        invalid_strings = {
            '', 'null', 'none', 'nan', 'na', 'n/a', 'not available', 
            'not found', 'empty', 'no data', 'undefined', 'missing',
            '0', '0.0', 'error', 'failed'
        }
        return value_clean not in invalid_strings and len(value.strip()) > 0
    
    if isinstance(value, (int, float)):
        # Check for NaN
        if isinstance(value, float) and value != value:  # NaN check
            return False
        # Zero is considered invalid for most use cases
        return value != 0
    
    if isinstance(value, (list, dict)):
        return len(value) > 0
    
    return True  # Other types are considered valid

def normalize_result(result_str: str) -> str:
    """Normalize result string for comparison"""
    if not result_str:
        return ""
    
    # Remove extra whitespace and normalize
    normalized = re.sub(r'\s+', ' ', str(result_str).strip())
    
    # Try to parse and re-serialize JSON for consistent formatting
    try:
        parsed = json.loads(normalized)
        return json.dumps(parsed, sort_keys=True, separators=(',', ':'))
    except:
        # If not JSON, just return normalized string
        return normalized


def results_are_equivalent(result1: str, result2: str, tolerance: dict = None) -> bool:
    """
    Compare two results with intelligent tolerance for different data types
    
    Args:
        result1, result2: Result strings to compare
        tolerance: Dict with tolerance settings
    """
    if tolerance is None:
        tolerance = {
            'numerical_tolerance': 2,  # 1% relative tolerance for numbers
            'absolute_tolerance': 0.001,   # Absolute tolerance for small numbers
            'ignore_whitespace': True,
            'ignore_case_in_strings': False,
            'base64_similarity_threshold': 0.95  # For image comparisons
        }
    
    # Quick exact match check first
    if normalize_result(result1) == normalize_result(result2):
        return True
    
    # Try to parse both as JSON and compare intelligently
    try:
        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)
        
        return json_results_equivalent(parsed1, parsed2, tolerance)
        
    except json.JSONDecodeError:
        # Not JSON, do string comparison with tolerance
        return string_results_equivalent(result1, result2, tolerance)


def json_results_equivalent(obj1, obj2, tolerance: dict) -> bool:
    """Compare JSON objects with tolerance for numerical differences"""
    
    # Handle different types
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        return all(json_results_equivalent(obj1[key], obj2[key], tolerance) 
                  for key in obj1.keys())
    
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(json_results_equivalent(a, b, tolerance) 
                  for a, b in zip(obj1, obj2))
    
    elif isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
        return numbers_equivalent(float(obj1), float(obj2), tolerance)
    
    elif isinstance(obj1, str) and isinstance(obj2, str):
        return strings_equivalent(obj1, obj2, tolerance)
    
    else:
        return obj1 == obj2


def numbers_equivalent(num1: float, num2: float, tolerance: dict) -> bool:
    """Check if two numbers are equivalent within tolerance"""
    
    # Handle exact matches
    if num1 == num2:
        return True
    
    # Handle zero cases
    if abs(num1) < tolerance['absolute_tolerance'] and abs(num2) < tolerance['absolute_tolerance']:
        return True
    
    # Relative tolerance for larger numbers
    if max(abs(num1), abs(num2)) > tolerance['absolute_tolerance']:
        relative_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
        return relative_diff <= tolerance['numerical_tolerance']
    
    # Absolute tolerance for small numbers
    return abs(num1 - num2) <= tolerance['absolute_tolerance']


def strings_equivalent(str1: str, str2: str, tolerance: dict) -> bool:
    """Check if two strings are equivalent with various tolerances"""
    
    # Base64 image comparison (common in data visualization tasks)
    if str1.startswith('data:image/') and str2.startswith('data:image/'):
        return base64_images_similar(str1, str2, tolerance['base64_similarity_threshold'])
    
    # Extract and compare numbers within strings
    nums1 = re.findall(r'-?\d+\.?\d*', str1)
    nums2 = re.findall(r'-?\d+\.?\d*', str2)
    
    if nums1 and nums2 and len(nums1) == len(nums2):
        # If strings contain numbers, compare them with tolerance
        try:
            for n1, n2 in zip(nums1, nums2):
                if not numbers_equivalent(float(n1), float(n2), tolerance):
                    return False
            # If all numbers match, check the string structure
            str1_no_nums = re.sub(r'-?\d+\.?\d*', 'NUM', str1)
            str2_no_nums = re.sub(r'-?\d+\.?\d*', 'NUM', str2)
            return str1_no_nums.strip() == str2_no_nums.strip()
        except ValueError:
            pass
    
    # Regular string comparison
    if tolerance['ignore_whitespace']:
        str1 = re.sub(r'\s+', ' ', str1.strip())
        str2 = re.sub(r'\s+', ' ', str2.strip())
    
    if tolerance['ignore_case_in_strings']:
        return str1.lower() == str2.lower()
    
    return str1 == str2


def base64_images_similar(img1: str, img2: str, threshold: float) -> bool:
    """Compare base64 encoded images by size similarity (rough approximation)"""
    try:
        # Extract base64 data
        data1 = img1.split(',', 1)[1] if ',' in img1 else img1
        data2 = img2.split(',', 1)[1] if ',' in img2 else img2
        
        # Compare lengths as a rough similarity measure
        len1, len2 = len(data1), len(data2)
        if len1 == 0 or len2 == 0:
            return len1 == len2
        
        similarity = min(len1, len2) / max(len1, len2)
        return similarity >= threshold
        
    except Exception:
        return False


def string_results_equivalent(result1: str, result2: str, tolerance: dict) -> bool:
    """Compare non-JSON string results with tolerance"""
    
    # Try to find and compare all numbers in the strings
    nums1 = re.findall(r'-?\d+\.?\d*', result1)
    nums2 = re.findall(r'-?\d+\.?\d*', result2)
    
    if nums1 and nums2:
        try:
            # Convert to floats and compare with tolerance
            floats1 = [float(n) for n in nums1]
            floats2 = [float(n) for n in nums2]
            
            if len(floats1) == len(floats2):
                for f1, f2 in zip(floats1, floats2):
                    if not numbers_equivalent(f1, f2, tolerance):
                        return False
                return True
        except ValueError:
            pass
    
    # Fallback to string comparison
    return strings_equivalent(result1, result2, tolerance)


async def generate_and_execute_code_single_attempt(prompt: str, work_folder: str, html_file_path: str, 
                                                 question: str, max_attempts: int, round_num: int):
    """Single attempt at generating and executing code (used within verification)"""
    
    attempt = 0
    last_error = None
    previous_code = ""
    
    while attempt < max_attempts:
        attempt += 1
        print(f"   Attempt {attempt} of round {round_num}")
        
        if attempt == 1:
            code_response = model.generate_content(prompt)
        else:
            refinement_prompt = f"""
The previous code failed. Fix these issues and generate working Python code:

ERROR: {last_error}

ORIGINAL QUESTION: {question}
HTML FILE: {html_file_path}

REQUIREMENTS:
- Fix the error above
- Extract real data from the HTML file  
- Never use placeholder/hardcoded values
- Import all needed libraries
- Handle edge cases (missing data, empty tables, etc.)
- Return results in exact format requested
- For missing data, use appropriate defaults (empty string, 0, empty list)
- BE CONSISTENT with data extraction methodology

PREVIOUS CODE (that failed):
{previous_code}

Generate the corrected Python code:
"""
            code_response = model.generate_content(refinement_prompt)
        
        # Extract and execute code
        generated_code = extract_code_blocks(code_response.text)
        code_file = save_code_to_file(generated_code, work_folder, f"round_{round_num}_attempt_{attempt}")
        
        try:
            result = subprocess.run(
                ["python3", code_file],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_folder
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if result.returncode == 0 and stdout:
                print(f"   ✅ Round {round_num}, Attempt {attempt} successful")
                return {
                    "question": question,
                    "answer": stdout,
                    "code": generated_code,
                    "round": round_num,
                    "attempt": attempt
                }
            else:
                last_error = stderr or "No output generated"
                previous_code = generated_code
                print(f"   ❌ Error: {last_error[:100]}...")
                
        except subprocess.TimeoutExpired:
            last_error = "Code execution timeout (60 seconds)"
            previous_code = generated_code
        except Exception as e:
            last_error = f"Execution error: {str(e)}"
            previous_code = generated_code
    
    return {
        "question": question,
        "error": f"Round {round_num} failed after {max_attempts} attempts. Last error: {last_error}",
        "round": round_num,
        "final_code": generated_code if 'generated_code' in locals() else "No code generated"
    }


def extract_data_preview(soup, data_type: str, max_items: int = 5):
    """Extract a preview of the data structure to help LLM understand the page"""
    
    preview = {"type": data_type, "sample_data": []}
    
    if data_type == "table":
        # Find tables and extract sample rows
        tables = soup.find_all("table")
        for i, table in enumerate(tables[:2]):  # Check first 2 tables
            rows = table.find_all("tr")[:max_items]
            table_preview = []
            for row in rows:
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                if cells:  # Only add non-empty rows
                    table_preview.append(cells)
            if table_preview:
                preview["sample_data"].append(f"Table {i+1}: {table_preview}")
    
    elif data_type == "list":
        # Find lists
        lists = soup.find_all(["ul", "ol"])
        for i, lst in enumerate(lists[:2]):
            items = [li.get_text(strip=True) for li in lst.find_all("li")[:max_items]]
            if items:
                preview["sample_data"].append(f"List {i+1}: {items}")
    
    else:
        # General text extraction
        # Look for common content containers
        containers = soup.find_all(["div", "section", "article", "main"])[:3]
        for i, container in enumerate(containers):
            text = container.get_text(strip=True)[:200]  # First 200 chars
            if text:
                preview["sample_data"].append(f"Container {i+1}: {text}...")
    
    return preview


async def enhanced_file_analysis_with_verification(question: str, detection_info: dict, saved_files: dict, work_folder: str):
    """Enhanced file analysis with result verification - handles CSV, PDF, images, ZIP, etc."""
    
    print(f"\n📁 Starting file-based analysis...")
    print(f"Available files: {list(saved_files.keys())}")
    
    # Step 1: Analyze uploaded files and create file preview
    file_analysis = analyze_uploaded_files(saved_files, work_folder)
    print(file_analysis)
    
    # Step 2: Generate and verify results with file-based approach
    return await generate_and_verify_file_results(
        question=question,
        detection_info=detection_info,
        file_analysis=file_analysis,
        saved_files=saved_files,
        work_folder=work_folder,
        max_attempts=6,
        verification_rounds=5
    )


def analyze_uploaded_files(saved_files: dict, work_folder: str) -> dict:
    """Analyze uploaded files to understand their structure and content"""
    
    analysis = {
        "files": [],
        "summary": "",
        "data_preview": {},
        "file_types": set()
    }
    
    for filename, filepath in saved_files.items():
        if filename.lower() == "questions.txt":
            continue  # Skip the questions file
            
        file_info = {
            "name": filename,
            "path": filepath,
            "size": os.path.getsize(filepath),
            "type": get_file_type(filename),
            "preview": None
        }
        
        # Extract preview based on file type
        try:
            if file_info["type"] == "csv":
                file_info["preview"] = preview_csv(filepath)
            elif file_info["type"] == "excel":
                file_info["preview"] = preview_excel(filepath)
            elif file_info["type"] == "pdf":
                file_info["preview"] = preview_pdf(filepath)
            elif file_info["type"] == "jpg":
                file_info["preview"] = preview_image(filepath)
            elif file_info["type"] == "image":
                print("got the image file bro")
                file_info["preview"] = preview_image(filepath)  
            elif file_info["type"] == "jpeg":
                file_info["preview"] = preview_image(filepath)      
            elif file_info["type"] == "zip":
                file_info["preview"] = preview_zip(filepath, work_folder)
            elif file_info["type"] == "json":
                file_info["preview"] = preview_json(filepath)
            elif file_info["type"] == "text":
                file_info["preview"] = preview_text(filepath)
                
        except Exception as e:
            file_info["preview"] = f"Error reading file: {str(e)}"
            
        analysis["files"].append(file_info)
        analysis["file_types"].add(file_info["type"])
    
    # Create summary
    file_type_counts = {}
    for file_info in analysis["files"]:
        file_type = file_info["type"]
        file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
    
    analysis["summary"] = f"Found {len(analysis['files'])} files: " + ", ".join([f"{count} {ftype}" for ftype, count in file_type_counts.items()])
    
    return analysis


def get_file_type(filename: str) -> str:
    """Determine file type from extension"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    type_mapping = {
        'csv': 'csv',
        'xlsx': 'excel', 'xls': 'excel',
        'pdf': 'pdf',
        'jpg': 'image', 'jpeg': 'image', 'png': 'image', 'gif': 'image', 'bmp': 'image',
        'zip': 'zip', 'rar': 'archive', '7z': 'archive',
        'json': 'json',
        'txt': 'text', 'md': 'text',
        'parquet': 'parquet',
        'sql': 'sql'
    }
    
    return type_mapping.get(ext, 'unknown')


def preview_csv(filepath: str, max_rows: int = 5) -> dict:
    """Preview CSV file content"""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(max_rows).to_dict('records'),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    except Exception as e:
        return {"error": str(e)}


def preview_excel(filepath: str, max_rows: int = 5) -> dict:
    """Preview Excel file content"""
    try:
        import pandas as pd
        # Read first sheet
        df = pd.read_excel(filepath)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(max_rows).to_dict('records'),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    except Exception as e:
        return {"error": str(e)}


def preview_pdf(filepath: str, max_chars: int = 500) -> dict:
    """Preview PDF content (native text extraction with OCR fallback if needed)."""
    try:
        from PyPDF2 import PdfReader
        import pdfplumber
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image, ImageOps

        text = ""
        warnings = []

        # --- Step 1: Try pdfplumber (best for text layout) ---
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages[:5]:  # preview first 5 pages
                    text += page.extract_text() or ""
            text = text.strip()
        except Exception as e:
            warnings.append(f"pdfplumber failed: {e}")

        # --- Step 2: Fallback: PyPDF2 ---
        if not text:
            try:
                reader = PdfReader(filepath)
                for page in reader.pages[:5]:
                    text += page.extract_text() or ""
                text = text.strip()
            except Exception as e:
                warnings.append(f"PyPDF2 failed: {e}")

        # --- Step 3: OCR fallback if still empty ---
        if not text:
            try:
                images = convert_from_path(filepath, dpi=300)
                for im in images[:3]:  # OCR first 3 pages
                    gray = ImageOps.grayscale(im)
                    text += pytesseract.image_to_string(gray, lang="eng")
                text = text.strip()
                if not text:
                    warnings.append("OCR produced no text")
            except Exception as e:
                warnings.append(f"OCR fallback failed: {e}")
            print({
            "type": "PDF document",
            "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "preview": text[:max_chars],
            "truncated": len(text) > max_chars,
            "warnings": warnings
        })
        return {
            "type": "PDF document",
            "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "preview": text[:max_chars],
            "truncated": len(text) > max_chars,
            "warnings": warnings
        }
    except Exception as e:
        return {"error": str(e)}



def preview_image(filepath: str, max_chars: int = 300) -> dict:
    """Preview image content (OCR text extraction)."""
    print("Preview image content (OCR text extraction)")
    try:
        import pytesseract
        from PIL import Image, ImageOps

        im = Image.open(filepath)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im = ImageOps.grayscale(im)
        im = ImageOps.autocontrast(im)

        text = pytesseract.image_to_string(im, lang="eng").strip()
        print({
            "type": "Image file",
            "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "path": filepath,
            "preview": text[:max_chars],
            "truncated": len(text) > max_chars
        })
        return {
            "type": "Image file",
            "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "path": filepath,
            "preview": text[:max_chars],
            "truncated": len(text) > max_chars
        }
    except Exception as e:
        return {"error": str(e)}

def preview_zip(filepath: str, work_folder: str) -> dict:
    """Preview ZIP file contents"""
    try:
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            file_list = zip_ref.namelist()[:10]  # First 10 files
            # Extract to work folder for analysis
            extract_folder = os.path.join(work_folder, "extracted")
            os.makedirs(extract_folder, exist_ok=True)
            zip_ref.extractall(extract_folder)
            
        return {
            "total_files": len(zip_ref.namelist()),
            "sample_files": file_list,
            "extracted_to": extract_folder,
            "note": "ZIP contents extracted for analysis"
        }
    except Exception as e:
        return {"error": str(e)}


def preview_json(filepath: str, max_items: int = 5) -> dict:
    """Preview JSON file content"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return {
                "type": "JSON Array",
                "length": len(data),
                "sample_items": data[:max_items]
            }
        elif isinstance(data, dict):
            return {
                "type": "JSON Object", 
                "keys": list(data.keys())[:max_items],
                "sample_data": {k: v for i, (k, v) in enumerate(data.items()) if i < max_items}
            }
        else:
            return {"type": "JSON", "content": str(data)[:200]}
            
    except Exception as e:
        return {"error": str(e)}


def preview_text(filepath: str, max_chars: int = 300) -> dict:
    """Preview text file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(max_chars)
        return {
            "type": "Text file",
            "size": os.path.getsize(filepath),
            "preview": content,
            "truncated": os.path.getsize(filepath) > max_chars
        }
    except Exception as e:
        return {"error": str(e)}


async def generate_and_verify_file_results(
    question: str,
    detection_info: dict,
    file_analysis: dict,
    saved_files: dict,
    work_folder: str,
    max_attempts: int = 6,
    verification_rounds: int = 5
):
    """Generate and return results for file-based analysis"""

    print(f"\n🔍 Starting simplified file analysis process...")

    for round_num in range(verification_rounds):
        print(f"\n--- File Analysis Round {round_num + 1} ---")

        # Generate result for this round
        round_result = await generate_and_execute_file_code_single_attempt(
            prompt=f"""
You are a Python data analysis expert. Analyze the provided files and answer this question:

QUESTION: {question}

AVAILABLE FILES:
{json.dumps({info["name"]: info["preview"] for info in file_analysis["files"] if info["name"].lower() != "questions.txt"}, indent=2)}
FILE ANALYSIS SUMMARY:
{file_analysis["summary"]}

WORK DIRECTORY: {work_folder}
All uploaded files are available in this directory.

FILE DETAILS:
{json.dumps([{
    "filename": info["name"],
    "filepath": info["path"], 
    "type": info["type"],
    "size": info["size"],
    "preview": info["preview"]
} for info in file_analysis["files"]], indent=2)}

REQUIREMENTS:
1. Load and analyze the appropriate files from the work directory
2. Use the correct libraries for each file type:
   - CSV/Excel: pandas
   - Images:  OpenCV, matplotlib, PIL
   - PDF: PyPDF2, pdfplumber
   - ZIP: zipfile (already extracted if needed)
   - JSON: json library
3. Perform all required calculations/analysis based on the question
4. Handle missing or corrupted data gracefully
5. Return results in the exact format requested in the question
6. For visualizations, save as PNG and encode as base64 data URI

CRITICAL RULES:
- Import all necessary libraries at the top
- Use the actual file paths provided above
- Never hardcode answers - compute everything from the actual data
- Print only the final result (JSON format if requested)
- Be consistent in your data processing methodology
- Handle different file types appropriately
- For statistical calculations, use sufficient precision
-- I want final output as json and dont give array or list as output give only json like this: 
    {{"question 1" : "answer 1 ","question 2 " : "answer 2 "}} 

Write the complete Python code to solve this question:
""",
            work_folder=work_folder,
            question=question,
            max_attempts=max_attempts,
            round_num=round_num + 1
        )

        # If execution succeeded and output is valid → parse and return
        if not round_result.get("error"):
            result_answer = round_result.get("answer", "").strip()
            if result_answer and result_answer != "0":
                print(f"✅ Round {round_num + 1} successful, parsing and returning result")
                
                # Try to parse as JSON and return the parsed object directly
                try:
                    parsed_result = json.loads(result_answer)
                    # Return the parsed JSON object directly, not wrapped in "answer"
                    return parsed_result
                except json.JSONDecodeError:
                    # If not valid JSON, still return meaningful data
                    print(f"⚠️ Result is not valid JSON, returning as result field: {result_answer[:100]}...")
                    return {"result": result_answer}
            else:
                print(f"⚠️ Round {round_num + 1} produced invalid output (empty/null/0), retrying...")
        else:
            print(f"❌ Round {round_num + 1} failed: {round_result.get('error', 'Unknown error')}")

    # If all rounds fail, return error
    return {
        "question": question,
        "error": f"All {verification_rounds} file analysis rounds failed or returned invalid output",
        "verification_status": "FAILED",
        "analysis_type": "FILE_BASED",
        "files_analyzed": list(saved_files.keys()),
    }


async def generate_and_execute_file_code_single_attempt(
    prompt: str,
    work_folder: str,
    question: str,
    max_attempts: int,
    round_num: int
):
    """Single attempt at generating and executing file analysis code"""

    attempt = 0
    last_error = None
    previous_code = ""

    while attempt < max_attempts:
        attempt += 1
        print(f"   File Analysis Attempt {attempt} of round {round_num}")

        if attempt == 1:
            code_response = model.generate_content(prompt)
        else:
            refinement_prompt = f"""
The previous file analysis code failed. Fix these issues and generate working Python code:

ERROR: {last_error}

ORIGINAL QUESTION: {question}
WORK DIRECTORY: {work_folder}

REQUIREMENTS:
- Fix the error above
- Use the correct file paths from the work directory
- Import all necessary libraries for file processing
- Handle different file types appropriately (CSV, Excel, PDF, images, etc.)
- Never use placeholder/hardcoded values
- Extract real data from the uploaded files
- Handle edge cases (missing files, corrupted data, etc.)
- Return results in exact format requested

PREVIOUS CODE (that failed):
{previous_code}

Generate the corrected Python code for file analysis:
"""
            code_response = model.generate_content(refinement_prompt)

        # Extract and execute code
        generated_code = extract_code_blocks(code_response.text)
        code_file = save_code_to_file(
            generated_code, work_folder,
            f"file_analysis_round_{round_num}_attempt_{attempt}"
        )

        try:
            result = subprocess.run(
                ["python3", code_file],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_folder
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0 and stdout and stdout != "0":
                print(f"   ✅ File Analysis Round {round_num}, Attempt {attempt} successful")
                # Return just the clean answer, no wrapper
                return {
                    "answer": stdout,
                }
            else:
                last_error = stderr or "No valid output generated"
                previous_code = generated_code
                print(f"   ❌ Error: {last_error[:100]}...")

        except subprocess.TimeoutExpired:
            last_error = "Code execution timeout (60 seconds)"
            previous_code = generated_code
        except Exception as e:
            last_error = f"Execution error: {str(e)}"
            previous_code = generated_code

    return {
        "question": question,
        "error": f"File Analysis Round {round_num} failed after {max_attempts} attempts. Last error: {last_error}",
        "round": round_num,
        "final_code": generated_code if 'generated_code' in locals() else "No code generated"
    }


def extract_json(text: str) -> dict:
    """Improved JSON extraction from LLM response with better error handling"""
    # Remove markdown fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON object pattern
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        
        # Fix common JSON issues
        # 1. Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"([^"]*)"([^,:}\]]*)"', r'"\1\"\2"', json_str)
        
        # 2. Convert Python lists to JSON arrays in steps
        json_str = re.sub(r'"steps":\s*\[([^\]]+)\]', 
                         lambda m: '"steps": [' + ', '.join(f'"{step.strip()}"' if not step.strip().startswith('"') else step.strip() 
                                                           for step in m.group(1).split('", "')) + ']', 
                         json_str)
        
        # 3. Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error after cleanup: {e}")
            print(f"Cleaned JSON: {json_str[:500]}...")
    
    # Final fallback - try to extract key information manually
    print("Using manual JSON extraction as fallback")
    
    # Extract URLs manually
    url_matches = re.findall(r'https?://[^\s",\]]+', text)
    
    # Extract scraping decision
    scraping = "yes" if "scraping" in text.lower() and ("yes" in text.lower() or url_matches) else "no"
    
    return {
        "scraping": scraping,
        "urls": url_matches if url_matches else [],
        "data_type": "table", 
        "target_elements": ["table", "tr", "td"],
        "steps": [
            "Parse HTML file with BeautifulSoup",
            "Find and extract table data", 
            "Clean and process the data",
            "Perform required calculations",
            "Generate visualizations if needed",
            "Return results in JSON format"
        ]
    }


def extract_code_blocks(text: str) -> str:
    """Extract Python code from LLM response"""
    # Look for fenced code blocks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no fenced blocks, look for code-like patterns
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Start of code indicators
        if any(line.strip().startswith(x) for x in ['import ', 'from ', 'def ', 'class ', 'if __name__']):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # Fallback - return entire text
    return text.strip()


def save_code_to_file(code: str, folder: str, filename_prefix: str = "scraper") -> str:
    """Save code to a file and return the path"""
    filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.py"
    filepath = os.path.join(folder, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    
    return filepath


def is_valid_result(result: str) -> bool:
    """
    Check if a result is valid (not empty, null, zero, or meaningless)
    
    Args:
        result: The result string to validate
        
    Returns:
        bool: True if result is valid, False otherwise
    """
    if not result or result.strip() == "":
        return False
    
    # Normalize the result
    normalized = str(result).strip().lower()
    
    # Check for common invalid values
    invalid_values = {
        "null", "none", "na", "n/a", "undefined", "nan", 
        "error", "failed", "0", "0.0", "[]", "{}", 
        "empty", "no data", "not found", "not available"
    }
    
    if normalized in invalid_values:
        return False
    
    # Check for JSON with empty/null values
    try:
        parsed = json.loads(result)
        
        # Check different JSON structures
        if isinstance(parsed, dict):
            # Empty dict or dict with all null/empty values
            if not parsed:
                return False
            return any(is_value_meaningful(v) for v in parsed.values())
            
        elif isinstance(parsed, list):
            # Empty list or list with all null/empty values
            if not parsed:
                return False
            return any(is_value_meaningful(v) for v in parsed)
            
        else:
            # Single value
            return is_value_meaningful(parsed)
            
    except json.JSONDecodeError:
        # Not JSON, check as string
        pass
    
    # For non-JSON strings, check if they contain meaningful content
    # Remove common filler words and check length
    meaningful_content = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', normalized)
    meaningful_content = re.sub(r'\s+', ' ', meaningful_content).strip()
    
    # Must have some meaningful content (more than just numbers/punctuation)
    has_letters = bool(re.search(r'[a-z]', meaningful_content))
    has_numbers = bool(re.search(r'\d', meaningful_content))
    
    # Valid if it has letters, or meaningful numbers (not just "0")
    if has_letters:
        return len(meaningful_content) > 3
    elif has_numbers:
        # Check if numbers are meaningful (not just zeros)
        numbers = re.findall(r'-?\d+\.?\d*', result)
        return any(float(n) != 0 for n in numbers if n)
    
    return False


def is_value_meaningful(value) -> bool:
    """Check if a single value is meaningful"""
    if value is None:
        return False
    
    if isinstance(value, str):
        return value.strip() not in ["", "null", "none", "na", "n/a", "undefined", "nan", "error", "failed", "0", "empty"]
    
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and (value != value))  # Check for NaN
    
    if isinstance(value, (list, dict)):
        return len(value) > 0
    
    return True

def get_package_name_mapping() -> dict:
    """
    Map import names to pip package names (since they're sometimes different)
    
    Returns:
        Dictionary mapping import names to pip package names
    """
    return {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'bs4': 'beautifulsoup4',
        'requests': 'requests',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'scipy': 'scipy',
        'openpyxl': 'openpyxl',
        'xlrd': 'xlrd',
        'PyPDF2': 'PyPDF2',
        'pdfplumber': 'pdfplumber',
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'wordcloud': 'wordcloud',
        'textstat': 'textstat',
        'nltk': 'nltk',
        'spacy': 'spacy',
        'transformers': 'transformers',
        'torch': 'torch',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'statsmodels': 'statsmodels',
        # Add more mappings as needed
    }


def extract_missing_modules_from_error(error_text: str) -> list:
    """
    Extract missing module names from Python error messages
    
    Args:
        error_text: The stderr output from failed Python execution
        
    Returns:
        List of missing module names
    """
    missing_modules = []
    
    # Common error patterns for missing modules
    patterns = [
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        r"ImportError: No module named ['\"]([^'\"]+)['\"]", 
        r"ImportError: cannot import name ['\"][^'\"]+['\"] from ['\"]([^'\"]+)['\"]",
        r"from ([a-zA-Z_][a-zA-Z0-9_]*) import.*ModuleNotFoundError",
        r"import ([a-zA-Z_][a-zA-Z0-9_]*).*ModuleNotFoundError"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, error_text)
        for match in matches:
            # Get the base module name (before any dots)
            base_module = match.split('.')[0]
            missing_modules.append(base_module)
    
    return list(set(missing_modules))  # Remove duplicates


def install_package(package_name: str) -> tuple:
    """
    Install a package using pip
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        print(f"📦 Installing package: {package_name}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package_name}")
            return True, f"Successfully installed {package_name}"
        else:
            error_msg = result.stderr.strip()
            print(f"❌ Failed to install {package_name}: {error_msg}")
            return False, f"Failed to install {package_name}: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout while installing {package_name}"
    except Exception as e:
        return False, f"Exception while installing {package_name}: {str(e)}"


def install_missing_packages_from_error(error_text: str) -> tuple:
    """
    Install packages that are missing based on error messages
    
    Args:
        error_text: The stderr from failed code execution
        
    Returns:
        Tuple of (any_installed: bool, installation_report: dict)
    """
    missing_modules = extract_missing_modules_from_error(error_text)
    
    if not missing_modules:
        return False, {"message": "No missing modules detected in error"}
    
    package_mapping = get_package_name_mapping()
    installation_report = {
        "missing_modules_detected": missing_modules,
        "successfully_installed": [],
        "failed_to_install": [],
        "installation_errors": []
    }
    
    print(f"🔍 Detected missing modules from error: {missing_modules}")
    
    any_installed = False
    
    for module_name in missing_modules:
        # Get the pip package name
        package_name = package_mapping.get(module_name, module_name)
        
        # Try to install the package
        success, message = install_package(package_name)
        
        if success:
            installation_report["successfully_installed"].append(module_name)
            any_installed = True
        else:
            installation_report["failed_to_install"].append(module_name)
            installation_report["installation_errors"].append(f"{module_name} ({package_name}): {message}")
    
    print(f"📊 Installation Summary:")
    print(f"   Successfully installed: {installation_report['successfully_installed']}")
    print(f"   Failed to install: {installation_report['failed_to_install']}")
    
    return any_installed, installation_report
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
