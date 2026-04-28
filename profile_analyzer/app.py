from fastapi import FastAPI, BackgroundTasks, Request
from google.cloud import tasks_v2
import os
import json
import asyncio
import random
import logging
from typing import List, Callable, TypeVar, Tuple, Type, Dict, Any
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from asyncio import Semaphore
from openai import AsyncOpenAI

from google import genai
from google.genai import types
from google.cloud import storage
from google.protobuf import duration_pb2
import traceback


app = FastAPI()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar("T")


# Pull the three variables dynamically from the environment
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-gcs-bucket-name")
QUEUE = os.environ.get("QUEUE_NAME", "profile-analyzer-queue")

# Location can remain hardcoded since the queue and bucket are built here
LOCATION = "us-central1" 
    
def log_telemetry(username: str, kickoff_timestamp: str, stage: str, status: str):
    """
    Outputs a structured JSON log that Google Cloud Run automatically 
    parses into jsonPayload, making it easy to sink to BigQuery.
    """
    log_data = {        
        "telemetry_log": True,                 
        "username": username,
        "kickoff_timestamp": kickoff_timestamp,
        
        "stage": stage,
        "status": status
    }
        
    # converts it to a structured 'jsonPayload' in Google Cloud Logging.
    print(json.dumps(log_data), flush=True)

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to {destination_blob_name}")
    
def check_gcs_file_exists(bucket_name: str, file_path: str) -> bool:
    """Returns True if the file exists in the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return bucket.blob(file_path).exists()

def create_lock_file(bucket_name: str, username: str):
    """Creates a dummy processing.txt file to act as a lock."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"results/{username}/processing.txt")
    blob.upload_from_string("processing")

def delete_lock_file(bucket_name: str, username: str):
    """Deletes the processing.txt file once the job is finished."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"results/{username}/processing.txt")
    if blob.exists():
        blob.delete()

def retry_parse(
    producer: Callable[[], Any],
    parser: Callable[[Any], T],
    retries: int = 2,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> T:
    """Retry mechanism for API calls and parsing."""
    last_error = None
    for attempt in range(retries + 1):
        raw = None
        try:
            raw = producer()
            return parser(raw)
        except retry_exceptions as e:
            last_error = e
            logger.warning(f"Parse attempt {attempt + 1} failed: {e}. Raw data: {raw}")         
            
    raise last_error

class XInfoSchema(BaseModel):
    Topic: str = Field(description="Topic")
    Category: str = Field(description="Best matched Category")
    Reasoning: str = Field(description="Reasoning")
    Examples: list = Field(description="Examples")
    Perspective: list = Field(description="Perspective")


class ProfileAnalyzerWorkflow:
    """Orchestrates the user profiling, info gathering, and report generation."""
    
    def __init__(self, x_api_key: str, x_base_url: str, kickoff_ts: str):
        # Initialize clients once for the lifecycle of the workflow
        self.x_client = AsyncOpenAI(api_key=x_api_key, base_url=x_base_url)
        self.kickoff_ts = kickoff_ts 
        
        # Initialize Gemini clients (using Vertex AI as specified)
        self.genai_client = genai.Client(vertexai=True)
        self.genai_robust_client = genai.Client(
            vertexai=True,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    initial_delay=1.0,
                    attempts=5,
                    http_status_codes=[429],
                ),
                timeout=120 * 1000,
            ),
        )
        

    async def _send_x_request(self, sem: Semaphore, request: str, user: str, x_model: str) -> Any:
        """Send a single request to xAI with semaphore control."""
        async with sem:
            return await self.x_client.responses.parse(
                model=x_model,
                reasoning={"effort": "medium"},
                input=[{"role": "user", "content": request}],
                tools=[{
                    "type": "x_search",
                    "enable_image_understanding": True,
                    "allowed_x_handles": [user],
                    "from_date": "2026-01-01",
                    "to_date": "2026-02-27",
                }],
                text_format=XInfoSchema
            )

    async def gather_x_profile_info(self, user: str, context: str, topic: str, n: int = 3, x_model: str = 'grok-4-1-fast-reasoning') -> List[Dict]:
        """Retrieve and parse profile context from X."""
        prompt = f"""
        analyze all related the posts and comments from the account 
        {user} on X platform (formerly know as twitter). 
        Mainly focus on the related posts and comments. 
        based on their posts, comments and interactions determine the account holder's stance on:
        
        {context}      
        
        Please provide output as: 
            1- Topic: {topic} 
            2- Category: which category best fit the user (category number and category name)  
            3- Reasoning: Your Reasoning 
            4- Examples: Examples of their content support your reasoning (provide the full context, no urls). 
            5- Perspective: The account comprehensive perspective on this topic
        """
        
        sem = Semaphore(n)
        tasks = [self._send_x_request(sem, prompt, user, x_model) for _ in range(n)]
        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        parsed_responses = []
        for response in raw_responses:
            if isinstance(response, Exception):
                logger.error(f"X API request failed: {response}")
                continue
                
            try:         
                x_info = response.output_parsed
                parsed_responses.append({
                    'Topic': x_info.Topic,
                    'Category': x_info.Category,
                    'Reasoning': x_info.Reasoning,
                    'Perspective': x_info.Perspective,
                    'Example': x_info.Examples
                })
            except Exception as e:
                logger.error(f"Failed to parse X response: {e}")
                
        return parsed_responses

    def consolidate_responses(self, context: str, x_responses: List[Dict], model: str = 'gemini-3-flash-preview') -> Dict:
        """Consolidate multiple X model responses using Gemini."""
        prompt_intro = f"""
        We asked a large language model to determine someone's stance on:
        {context}

        Below, you will be provided with the original Source Text, followed by different responses 
        from models. Your task is to analyze these responses, resolve conflicts, and provide one accurate final response.

        Steps:
        1. Fact-Check against the Source.
        2. Resolve Conflicts.
        3. Consolidate valid points.
        4. Perspective Consolidation.
        5. Tone Generation.
        6. Final Output.
        """
        
        text_blocks = [
            f"model_id: {i}\nCategory: {r['Category']}\nReasoning: {r['Reasoning']}\nSource: {r['Example']}\n"
            for i, r in enumerate(x_responses)
        ]
        
        final_prompt = f"{prompt_intro}\nModels' responses:\n{''.join(text_blocks)}"
        
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "auditing": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "model_id": {"type": "STRING"},
                            "categorization": {"type": "BOOLEAN"},
                            "correct_category": {"type": "STRING"},
                            "why": {"type": "STRING"}
                        },
                        "required": ["model_id", "categorization", "correct_category", "why"]
                    }
                },
                "Final_Result": {
                    "type": "OBJECT",
                    "properties": {
                        "Category_Name": {"type": "STRING"},
                        "Category_Number": {"type": "INTEGER"},
                        "Reasoning": {"type": "STRING"},
                        "Perspective": {"type": "STRING"},
                        "Tone": {"type": "STRING"},
                        "Examples": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["Category_Name", "Category_Number", "Reasoning", "Perspective", "Tone", "Examples"]
                }
            },
            "required": ["auditing", "Final_Result"]
        }
        
        response = self.genai_robust_client.models.generate_content(
            model=model,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.0,
                seed=0,
                response_mime_type="application/json",
                response_schema=response_schema 
            )
        )
        return json.loads(response.text)

    def summarize_biography(self, full_result: Dict, model: str = "gemini-3.1-pro-preview") -> str:
        """Generate a full biographical summary based on all consolidated topics."""
        llm_context = """
        Based on the below information about the user write a "full and extensive biography" of this user. 
        Their biography should have the following sections:
          1- Islamic Republic Regime          
          2- Foreign countries
          3- Islamic Republic alternative
          4- Core ideology and political stand
        """
        
        for topic, data in full_result.items():
            context_data = data.get('Final_Result', {})
            context_str = '\n'.join(f"{k}:{v}" for k, v in context_data.items() if k != "Examples")
            llm_context += f"\n----\nOn topic: {topic}:\n{context_str}\n"

        response = self.genai_robust_client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=llm_context)])],
            config=types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=65535,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        )
        return response.text

    def calculate_score(self, full_result: Dict) -> float:
        """Calculate the Haramzade_Score based on mappings safely."""
       
        return 0

    def generate_infograph(self, bio: str, score: float, user: str) -> None:
        """Generate and save an infographic based on the biography and score."""
        prompt = f"""
        Create the Infograph demonstrating all aspects base on the below information. 
        Make sure their most dominant point of view is very clear in the Infograph.
        They have a score which is called "Alignment_Score" in the range of 0 to 100 it is {score:.2f}. 
        The Alignment_Score in the form of Speedometer (make sure the Speedometer show the accurate number) when        
        If the score is close to 100 use dark, gray and cold background for the Infograph.
        If the score is close to 0 use bright backhground for the Infograph.
        Do not draw or add the image of the user to the infograph.
        
        Bio:
        {bio}
        """

        try:
            image_response = self.genai_robust_client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    max_output_tokens=32768,
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="1K",
                        output_mime_type="image/png",
                    ),
                ),
            )
            
            for part in image_response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    filename = f"generated_image_{user}.png"
                    image.save(filename)
                    logger.info(f"Successfully saved infographic to {filename}")
        except Exception as e:
            logger.error(f"Failed to generate or save image: {e}")

    async def run_pipeline(self, questionnaire_file: str, user: str, area: str):
        """Main execution flow for reading topics, analyzing, and generating outputs."""
        logger.info(f"Starting pipeline for user: {user}")
        full_result = {}

        with open(questionnaire_file, "r") as f:
            content = f.read()

        blocks = [b.strip() for b in content.split("---") if b.strip()]

        for block in blocks:
            lines = block.splitlines()
            topic = lines[0].replace("Topic:", "").strip()
            context = f"{topic}\n" + "\n".join(lines[1:]).strip()
            
            logger.info(f"Processing Topic: {topic}")
            log_telemetry(user, self.kickoff_ts, stage=f"Processing Topic: {topic}", status="START")

            
            # 1. Gather info from X
            x_responses = await self.gather_x_profile_info(user, context, topic, n=3)
            if not x_responses:
                logger.warning(f"No valid X responses gathered for topic: {topic}")
                continue
            
            log_telemetry(user, self.kickoff_ts, stage=f"Processing Topic: {topic}", status="FINISH")
            log_telemetry(user, self.kickoff_ts, stage=f"Processing Topic: {topic} consolidation", status="START")

            # 2. Consolidate via Gemini
            consolidated = retry_parse(
                producer=lambda: self.consolidate_responses(context, x_responses),
                parser=lambda r: r, 
                retries=2,
                retry_exceptions=(json.JSONDecodeError, KeyError, TypeError)
                
            )
            log_telemetry(user, self.kickoff_ts, stage=f"Processing Topic: {topic} consolidation", status="FINISH")

            full_result[topic] = consolidated

        # 3. Save raw JSON results
        output_json = f"full_result_{user}.json"
        output_bio = f"bio_{user}.txt"
        output_score = f"score_{user}.txt"
        
        log_telemetry(user, self.kickoff_ts, stage=f"save result and bio locally", status="START")
        with open(output_json, "w") as f:
            json.dump(full_result, f, indent=4)
        logger.info(f"Saved consolidated results to {output_json}")

        # 4. Save bio results
        bio_text = self.summarize_biography(full_result)
        with open(output_bio, "w") as f:
                f.write(bio_text)
        logger.info(f"Saved bio to {output_json}")
        log_telemetry(user, self.kickoff_ts, stage=f"save result and bio locally", status="FINISH")

        # 5. Generate bio & score, then plot image
        log_telemetry(user, self.kickoff_ts, stage=f"calculate score", status="START")        
        final_score = self.calculate_score(full_result)
        log_telemetry(user, self.kickoff_ts, stage=f"calculate score", status="FINISH")
        log_telemetry(user, self.kickoff_ts, stage=f"save score locally", status="START")
        with open(output_score, "w") as f:
                f.write(str(final_score))
        logger.info(f"Calculated Score: {final_score:.2f}")
        log_telemetry(user, self.kickoff_ts, stage=f"save score locally", status="FINISH")
        
        log_telemetry(user, self.kickoff_ts, stage=f"generate infograph", status="START")
        self.generate_infograph(bio=bio_text, score=final_score, user=user)
        log_telemetry(user, self.kickoff_ts, stage=f"generate infograph", status="FINISH")
        logger.info("Pipeline completed successfully.")




@app.post("/analyze")
async def trigger_analysis(username: str):
    """
    1. The user calls this endpoint.
    2. We create a Google Cloud Task to do the heavy lifting.
    3. We return a response instantly.
    """


    # Step 1: Check if the final files already exist
    json_exists = check_gcs_file_exists(BUCKET_NAME, f"results/{username}/data.json")
    img_exists = check_gcs_file_exists(BUCKET_NAME, f"results/{username}/infograph.png")
    
    if json_exists or img_exists:
        return {
            "status": "completed",
            "username": username,
            "message": "Analysis already exists.",
            "data_url": f"https://storage.googleapis.com/{BUCKET_NAME}/results/{username}/data.json",
            "image_url": f"https://storage.googleapis.com/{BUCKET_NAME}/results/{username}/infograph.png"
        }

    # Step 2: Check if it is currently being processed
    if check_gcs_file_exists(BUCKET_NAME, f"results/{username}/processing.txt"):
        return {
            "status": "processing",
            "username": username,
            "message": "This profile is currently being analyzed by another process. Please check back in a few minutes."
        }

    # Step 3: It is brand new! Create the lock file and trigger the queue
    create_lock_file(BUCKET_NAME, username)

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(PROJECT_ID, LOCATION, QUEUE)

    # The URL of this very same Cloud Run service
    service_url = os.environ.get("SERVICE_URL", "https://your-cloud-run-url.a.run.app")
    
    timeout_duration = duration_pb2.Duration()
    timeout_duration.FromSeconds(1800)
    kickoff_ts = datetime.now(timezone.utc).isoformat()

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{service_url}/process-worker",
            "headers": {"Content-type": "application/json"},
            "body": json.dumps({"username": username,"kickoff_timestamp": kickoff_ts }).encode(),
        },
        "dispatch_deadline": timeout_duration 
    }
    
    # Send to the background queue
    client.create_task(request={"parent": parent, "task": task})

    return {
        "status": "queued",
        "username": username,
        "message": "Analysis started. Results will be saved to GCS bucket in ~20 minutes."
    }

@app.post("/process-worker")
async def process_worker(request: Request):
    """
    This endpoint is called by Google Cloud Tasks in the background.
    It can run for up to 30 minutes safely without dropping.
    """
    data = await request.json()
    username = data["username"]
    kickoff_ts = data["kickoff_timestamp"] 
    
    try:
        # 1. Run the script
        analyzer = ProfileAnalyzerWorkflow(
            x_api_key=os.environ["X_API_KEY"], 
            x_base_url="https://api.x.ai/v1",
            kickoff_ts=kickoff_ts
        )

        log_telemetry(
            username=username, 
            kickoff_timestamp=kickoff_ts, 
            stage="run_pipeline", 
            status="START"
        )

        await analyzer.run_pipeline("questionnaire.txt", username, "politics")
        
        log_telemetry(username, kickoff_ts, stage=f"run_pipeline", status="FINISH")

        log_telemetry(username, kickoff_ts, stage=f"save result in GCS", status="START")
        
        # 2. Upload the saved files to GCS    
        json_filename = f"full_result_{username}.json"
        if os.path.exists(json_filename):
            upload_to_gcs(
                json_filename, 
                BUCKET_NAME, 
                f"results/{username}/full_result.json"
            )
        else:
            logging.warning(f"JSON file {json_filename} was not generated. Skipping upload.")
        
        bio_filename = f"bio_{username}.txt"
        if os.path.exists(bio_filename):
            upload_to_gcs(
                bio_filename, 
                BUCKET_NAME, 
                f"results/{username}/bio.txt"
            )
        else:
            logging.warning(f"JSON file {bio_filename} was not generated. Skipping upload.")
        
        score_filename = f"score_{username}.txt"
        if os.path.exists(score_filename):
            upload_to_gcs(
                score_filename, 
                BUCKET_NAME, 
                f"results/{username}/score.txt"
            )
        else:
            logging.warning(f"JSON file {score_filename} was not generated. Skipping upload.")
        
        image_filename = f"generated_image_{username}.png"
        if os.path.exists(image_filename):
            upload_to_gcs(
                image_filename, 
                BUCKET_NAME, 
                f"results/{username}/infograph.png"
            )
        else:
            logging.warning(f"Image file {image_filename} was not generated. Skipping upload.")

        log_telemetry(username, kickoff_ts, stage=f"save result in GCS", status="FINISH")
    except Exception as e:
        # If ANYTHING goes wrong, we catch it here so the server doesn't crash
        logging.error(f"An unexpected error occurred for user {username}: {{e}}")
        logging.error(traceback.format_exc()) 
        
    finally:        
        # Whether it succeeds, fails, or crashes, the lock file is ALWAYS deleted!
        logging.info(f"Cleaning up: Deleting lock file for {username}")
        delete_lock_file(BUCKET_NAME, username)
        
    return {"status": "success"}