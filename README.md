# Mapping the Discourse: Multi-Agent GenAI Pipeline

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-GCP-4285F4)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

This repository contains the source code and deployment configuration for a highly scalable, multi-agent Generative AI pipeline designed for **quantitative macro-sociological research**. 

Originally developed to map the complex ideological factions during the 2026 Iran protests, this system uses a combination of LLMs (`grok-4-1-fast-reasoning`, `Gemini 3.1 Pro`, `Gemini 3.1 Flash`) and image generation (`Nano Banana`) to analyze public social media discourse. By enforcing structured outputs and parallel agentic reasoning, it extracts, fact-checks, and categorizes ideological stances across multiple interconnected topics, allowing researchers to cluster and analyze digital public squares at scale.

**Read the full architectural breakdown and case study here:** *(Link to your article)*

---

## ⚠️ Ethical Boundary & Intended Use
**This application is a quantitative research tool with absolutely zero intent or capacity for individual profiling, doxing, or identity unmasking.** It is designed to process public posts from pseudonymous accounts to generate *aggregated, macro-level data* (such as correlation matrices and ideological clustering). It should strictly be used for academic, sociological, or data science research.

---

## Repository Contents

* `app.py`: The core FastAPI application. It contains the asynchronous ingestion API (`/analyze`) and the heavy AI worker endpoint (`/process-worker`) designed to be triggered by Google Cloud Tasks.
* `questionnaires.txt`: A customizable template file used to define the specific political/sociological topics and categorization schemas the AI agents will use.
* `requirements.txt`: Python dependencies.
* `Dockerfile`: Containerization instructions for deploying the application to Google Cloud Run.

---

## System Architecture Overview

To handle the ~20-minute execution time per account and bypass standard HTTP timeouts, the system uses an asynchronous worker pattern on Google Cloud Platform (GCP):

1. **Ingestion (`/analyze`):** Receives the target account, checks Google Cloud Storage (GCS) for existing cache or active process locks, and enqueues a payload to Google Cloud Tasks.
2. **Parallel Extraction (Grok):** The worker pulls the topics from `questionnaires.txt`, using `asyncio.gather` to run three parallel searches and categorizations per topic.
3. **Consensus & Fact-Checking (Gemini 3.1 Pro):** Acts as a master adjudicator, reviewing the 3x parallel Grok outputs, fact-checking the reasoning, and resolving hallucinations.
4. **Synthesis & Visualization:** Gemini 3.1 Flash compiles a structured biography, a custom scoring function calculates alignment, and Nano Banana generates a visual summary infographic. Outputs are saved to GCS.

---

## Customizing the Research Topics (`questionnaires.txt`)

To protect the specific prompts used in the original Iran 2026 case study, the `questionnaires.txt` file in this repository has been abstracted. 

To use this pipeline for your own sociological research, you must define the topics and the mutually exclusive categories you want the AI to classify the accounts into. The LLM pipeline dynamically parses this file to build its Structured Output schemas.

**Format your `questionnaires.txt` exactly like this:**

    Topic: [Insert your first research topic, e.g., Universal Healthcare]
       Categorize their stance into exactly one of the following four categories:
       Category 1: [Instruction/Description of Pro-stance]   
       Category 2: [Instruction/Description of Anti-stance]
       Category 3: [Instruction/Description of Nuanced/Gray Area]
       Category 4: [Instruction/Description of Unrelated/Silent]

    ---
    Topic: [Insert your second research topic, e.g., Climate Change Policy]
       Categorize their stance into exactly one of the following four categories:
       Category 1: [Instruction/Description...]   
       Category 2: [Instruction/Description...]   
       Category 3: [Instruction/Description...]
       Category 4: [Instruction/Description...]

*(Note: Separate each topic block with `---` so the `app.py` parser can split them into asynchronous batches).*

---

## Local Development & Testing

**1. Clone the repository:**

    git clone https://github.com/yourusername/mapping-the-discourse.git
    cd mapping-the-discourse

**2. Install dependencies:**

    pip install -r requirements.txt

**3. Set Environment Variables:**
You will need API keys for Grok, Google Cloud (Vertex AI), and Nano Banana.

    export GROK_API_KEY="your_grok_key"
    export GCP_PROJECT_ID="your_project_id"
    export NANO_BANANA_API_KEY="your_nano_banana_key"
    export GCS_BUCKET_NAME="your_storage_bucket"

**4. Run the FastAPI server locally:**

    uvicorn app:app --host 0.0.0.0 --port 8080 --reload

---

## GCP Deployment (Cloud Run)

This application is containerized and optimized for Google Cloud Run, utilizing Cloud Tasks for orchestration.

**1. Build and push the Docker image:**

    gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/discourse-mapper

**2. Deploy to Cloud Run:**

    gcloud run deploy discourse-mapper \
      --image gcr.io/YOUR_PROJECT_ID/discourse-mapper \
      --platform managed \
      --timeout 3600 \
      --memory 4Gi \
      --allow-unauthenticated

*(Note: The `--timeout 3600` flag sets the timeout to 1 hour to easily accommodate the ~20-minute AI pipeline).*

**3. Configure Cloud Tasks:**
Create a Cloud Task queue and configure the `/analyze` endpoint to route payloads to the secure Cloud Run `/process-worker` URL. Use queue rate limiting to prevent HTTP 429 (Too Many Requests) errors from the external LLM APIs.

---

## Observability

The `app.py` script is instrumented with structured JSON logging. By default, any log containing `{"telemetry_log": true}` can be intercepted by a GCP Log Router Sink and sent directly to BigQuery. This allows researchers to monitor pipeline execution times, failure rates, and stage progressions without needing a dedicated SQL database.
