"""
Vercel-Compatible Interview Assistant Backend with Resume Processing
✅ Optimized for Vercel serverless deployment
✅ FastAPI with proper module structure
✅ REST API only (no WebSockets)
✅ Resume processing with PyPDF2 (Vercel-compatible)

IMPORTANT: This file should NOT have 'handler = Mangum(app)' at the bottom!
The handler is defined in api/index.py
"""

import os
import json
import time
import io
import base64
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Initialize Supabase client (lazy loading)
supabase_client = None

def get_supabase_client():
    """Lazy load Supabase client"""
    global supabase_client
    if supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Supabase initialization error: {e}")
    return supabase_client

# FastAPI app
app = FastAPI(
    title="Interview Assistant API",
    version="2.0.0",
    description="Vercel-compatible interview assistant with AI-powered Q&A"
)

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "gpt-4o-mini"

# ============================================================================
# PYDANTIC MODELS - INTERVIEW ASSISTANT
# ============================================================================

class TranscriptRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    stream_type: str = "candidate"

class TranscriptResponse(BaseModel):
    transcript: str
    confidence: float = 0.0
    stream_type: str

class QuestionProcessRequest(BaseModel):
    transcript: str
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None
    custom_style_prompt: Optional[str] = None

class QuestionProcessResponse(BaseModel):
    has_question: bool
    question: Optional[str] = None
    answer: Optional[str] = None
    processing_time: float = 0.0

class BatchTranscriptRequest(BaseModel):
    transcripts: List[str]
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None

# ============================================================================
# PYDANTIC MODELS - RESUME PROCESSING
# ============================================================================

class ProcessResumeRequest(BaseModel):
    persona_id: str

class ProcessResumeResponse(BaseModel):
    success: bool
    message: str
    persona_id: str
    summary: Optional[str] = None
    processing_time: float = 0.0

class BulkProcessResumesRequest(BaseModel):
    user_id: Optional[str] = None
    persona_ids: Optional[List[str]] = None

# ============================================================================
# RESPONSE STYLES & PROMPTS
# ============================================================================

RESPONSE_STYLES = {
    "concise": {
        "name": "Concise Professional",
        "prompt": """You are a concise interview assistant. Provide brief, professional answers in 2-3 sentences.
Focus on the core information without elaboration. Be direct and efficient."""
    },
    "detailed": {
        "name": "Detailed Professional",
        "prompt": """You are a detailed interview assistant. Provide comprehensive answers with:
- Clear explanation of the concept
- Relevant examples from experience
- Practical insights
Keep responses around 150 words, professional and well-structured."""
    },
    "storytelling": {
        "name": "Storytelling",
        "prompt": """You are an engaging interview assistant using storytelling techniques.
Structure answers using STAR format when appropriate:
- Situation: Set the context
- Task: Describe the challenge
- Action: Explain what you did
- Result: Share the outcome
Make responses compelling and memorable while remaining professional."""
    },
    "technical": {
        "name": "Technical Expert",
        "prompt": """You are a technical interview expert. Provide in-depth technical answers:
- Explain concepts clearly with proper terminology
- Include code examples when relevant
- Discuss trade-offs and best practices
Be thorough but avoid unnecessary jargon."""
    }
}

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant that processes conversation transcripts.

Your task:
1. Analyze the incoming transcript text
2. Extract the EXACT question being asked (keep original wording)
3. If a question is detected, return it in this EXACT format:
   QUESTION: [extracted question]
   ANSWER: [your answer]
4. If it's casual conversation or greetings, respond with: "SKIP"

Guidelines:
- DO NOT rephrase the question - extract it EXACTLY as asked
- Remove only conversational preamble
- Keep ALL technical terms and original phrasing
"""

# ============================================================================
# DEEPGRAM TRANSCRIPTION (REST API)
# ============================================================================

async def transcribe_audio_deepgram(audio_base64: str, language: str = "en") -> Dict[str, Any]:
    """Transcribe audio using Deepgram REST API"""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
    
    import httpx
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        url = "https://api.deepgram.com/v1/listen"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true",
            "filler_words": "false"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                content=audio_bytes
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Deepgram API error: {response.text}"
                )
            
            result = response.json()
            
            transcript = ""
            confidence = 0.0
            
            if result.get("results", {}).get("channels", []):
                alternatives = result["results"]["channels"][0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    confidence = alternatives[0].get("confidence", 0.0)
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "full_response": result
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# ============================================================================
# AI QUESTION PROCESSING
# ============================================================================

async def process_question_with_ai(
    transcript: str,
    settings: Dict[str, Any],
    persona_data: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Process transcript and generate answer using OpenAI"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    if not transcript or len(transcript.strip()) < 10:
        return {
            "has_question": False,
            "question": None,
            "answer": None,
            "message": "Transcript too short"
        }
    
    start_time = time.time()
    
    try:
        response_style_id = settings.get("selectedResponseStyleId", "concise")
        
        if custom_style_prompt:
            style_prompt = custom_style_prompt
        else:
            style_config = RESPONSE_STYLES.get(response_style_id, RESPONSE_STYLES["concise"])
            style_prompt = style_config["prompt"]
        
        system_prompt = QUESTION_DETECTION_PROMPT + "\n\n" + style_prompt
        
        if persona_data:
            system_prompt += f"""

CANDIDATE CONTEXT:
- Position: {persona_data.get('position', 'N/A')}
- Company: {persona_data.get('company_name', 'N/A')}
"""
            if persona_data.get('company_description'):
                system_prompt += f"- Company Description: {persona_data.get('company_description')}\n"
            if persona_data.get('job_description'):
                system_prompt += f"- Job Description: {persona_data.get('job_description')}\n"
            if persona_data.get('resume_text'):
                system_prompt += f"\nCANDIDATE RESUME:\n{persona_data.get('resume_text')}\n"
        
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nWhen providing code examples, use {prog_lang}."
        
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        model = settings.get("defaultModel", DEFAULT_MODEL)
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {transcript}"}
            ],
            temperature=0.5,
            max_tokens=400,
            timeout=20
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.upper() == "SKIP" or "SKIP" in result_text.upper():
            return {
                "has_question": False,
                "question": None,
                "answer": None,
                "message": "Not a question",
                "processing_time": time.time() - start_time
            }
        
        question = None
        answer = None
        
        if "QUESTION:" in result_text and "ANSWER:" in result_text:
            parts = result_text.split("ANSWER:", 1)
            question = parts[0].replace("QUESTION:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            question = transcript
            answer = result_text
        
        return {
            "has_question": True,
            "question": question,
            "answer": answer,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# ============================================================================
# RESUME PROCESSING HELPER FUNCTIONS
# ============================================================================

async def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text using PyPDF2"""
    try:
        import requests
        from PyPDF2 import PdfReader
        
        response = requests.get(pdf_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF: HTTP {response.status_code}"
            )
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        extracted_text = text.strip()
        
        if len(extracted_text) < 50:
            return ""
        
        return extracted_text
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF extraction error: {str(e)}"
        )


async def generate_resume_summary(text: str) -> str:
    """Use OpenAI to generate a concise resume summary"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    if not text or len(text.strip()) < 50:
        return "No meaningful content found in resume."
    
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""Summarize this resume in 3-4 sentences covering:
1. Current role/most recent position
2. Key technical skills and expertise areas
3. Notable achievements or years of experience
4. Educational background (if mentioned)

Resume text:
{text}"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            timeout=15
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI summarization error: {str(e)}"
        )

# ============================================================================
# API ENDPOINTS - CORE
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "Interview Assistant API",
        "platform": "Vercel",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "POST /api/transcribe",
            "process_question": "POST /api/process-question",
            "batch_process": "POST /api/batch-process",
            "transcribe_and_answer": "POST /api/transcribe-and-answer",
            "process_resume": "POST /api/process-resume",
            "process_resumes_bulk": "POST /api/process-resumes-bulk",
            "resume_status": "GET /api/resume-status/{persona_id}",
            "models": "/api/models/status",
            "styles": "/api/response-styles"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    supabase = get_supabase_client()
    return {
        "status": "healthy",
        "platform": "Vercel",
        "timestamp": time.time(),
        "services": {
            "openai": "configured" if OPENAI_API_KEY else "missing",
            "deepgram": "configured" if DEEPGRAM_API_KEY else "missing",
            "supabase": "configured" if supabase else "missing"
        }
    }

# ============================================================================
# API ENDPOINTS - TRANSCRIPTION
# ============================================================================

@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(request: TranscriptRequest):
    """Transcribe audio using Deepgram REST API"""
    try:
        result = await transcribe_audio_deepgram(
            request.audio_base64,
            request.language
        )
        
        return TranscriptResponse(
            transcript=result["transcript"],
            confidence=result["confidence"],
            stream_type=request.stream_type
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-question", response_model=QuestionProcessResponse)
async def process_question(request: QuestionProcessRequest):
    """Process interview question and generate answer"""
    try:
        result = await process_question_with_ai(
            request.transcript,
            request.settings,
            request.persona,
            request.custom_style_prompt
        )
        
        return QuestionProcessResponse(**result)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-process")
async def batch_process_questions(request: BatchTranscriptRequest):
    """Process multiple questions at once"""
    try:
        results = []
        
        for transcript in request.transcripts:
            result = await process_question_with_ai(
                transcript,
                request.settings,
                request.persona
            )
            results.append(result)
        
        return {
            "total": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-and-answer")
async def transcribe_and_answer(request: Request):
    """One-shot endpoint: Transcribe audio AND generate answer"""
    try:
        data = await request.json()
        
        audio_base64 = data.get("audio_base64")
        language = data.get("language", "en")
        settings = data.get("settings", {})
        persona = data.get("persona")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 required")
        
        transcription = await transcribe_audio_deepgram(audio_base64, language)
        transcript = transcription["transcript"]
        
        if not transcript:
            return {
                "success": False,
                "message": "No transcript generated",
                "transcript": "",
                "answer": None
            }
        
        result = await process_question_with_ai(
            transcript,
            settings,
            persona
        )
        
        return {
            "success": True,
            "transcript": transcript,
            "confidence": transcription["confidence"],
            "has_question": result["has_question"],
            "question": result.get("question"),
            "answer": result.get("answer"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - RESUME PROCESSING
# ============================================================================

@app.post("/api/process-resume", response_model=ProcessResumeResponse)
async def process_single_resume(request: ProcessResumeRequest):
    """Process a single resume: extract text and generate AI summary"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured"
        )
    
    start_time = time.time()
    
    try:
        response = supabase.table("personas").select(
            "id, user_id, company_name, position, resume_url, resume_text"
        ).eq("id", request.persona_id).single().execute()
        
        persona = response.data
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        if persona.get("resume_text") and len(str(persona.get("resume_text", "")).strip()) > 50:
            return ProcessResumeResponse(
                success=True,
                message="Resume already processed",
                persona_id=request.persona_id,
                summary=persona["resume_text"],
                processing_time=time.time() - start_time
            )
        
        if not persona.get("resume_url"):
            raise HTTPException(
                status_code=400,
                detail="No resume URL found for this persona"
            )
        
        text = await extract_text_from_pdf_url(persona["resume_url"])
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )
        
        summary = await generate_resume_summary(text)
        
        supabase.table("personas").update({
            "resume_text": summary
        }).eq("id", request.persona_id).execute()
        
        return ProcessResumeResponse(
            success=True,
            message="Resume processed successfully",
            persona_id=request.persona_id,
            summary=summary,
            processing_time=time.time() - start_time
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


@app.post("/api/process-resumes-bulk")
async def process_resumes_bulk(request: BulkProcessResumesRequest):
    """Process multiple resumes at once"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        personas_to_process = []
        
        if request.user_id:
            response = supabase.table("personas").select(
                "id, user_id, company_name, position, resume_url, resume_text"
            ).eq("user_id", request.user_id).execute()
            
            personas_to_process = [
                p for p in response.data
                if p.get("resume_url") and (
                    not p.get("resume_text") or
                    len(str(p.get("resume_text", "")).strip()) < 50
                )
            ]
            
        elif request.persona_ids:
            for persona_id in request.persona_ids:
                response = supabase.table("personas").select(
                    "id, user_id, company_name, position, resume_url, resume_text"
                ).eq("id", persona_id).single().execute()
                
                persona = response.data
                if persona and persona.get("resume_url"):
                    if not persona.get("resume_text") or len(str(persona.get("resume_text", "")).strip()) < 50:
                        personas_to_process.append(persona)
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either user_id or persona_ids"
            )
        
        if not personas_to_process:
            return {
                "success": True,
                "message": "No unprocessed resumes found",
                "total": 0,
                "processed": 0,
                "failed": 0,
                "results": []
            }
        
        max_resumes = 3
        if len(personas_to_process) > max_resumes:
            return {
                "success": False,
                "message": f"Too many resumes to process at once (limit: {max_resumes}). Use single endpoint instead.",
                "total": len(personas_to_process),
                "processed": 0,
                "failed": 0
            }
        
        results = []
        processed = 0
        failed = 0
        
        for persona in personas_to_process:
            try:
                text = await extract_text_from_pdf_url(persona["resume_url"])
                
                if not text:
                    failed += 1
                    results.append({
                        "persona_id": persona["id"],
                        "success": False,
                        "error": "Could not extract text"
                    })
                    continue
                
                summary = await generate_resume_summary(text)
                
                supabase.table("personas").update({
                    "resume_text": summary
                }).eq("id", persona["id"]).execute()
                
                processed += 1
                results.append({
                    "persona_id": persona["id"],
                    "success": True,
                    "summary_length": len(summary)
                })
                
            except Exception as e:
                failed += 1
                results.append({
                    "persona_id": persona["id"],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "message": f"Processed {processed} of {len(personas_to_process)} resumes",
            "total": len(personas_to_process),
            "processed": processed,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/resume-status/{persona_id}")
async def get_resume_status(persona_id: str):
    """Check if a resume has been processed"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        response = supabase.table("personas").select(
            "id, resume_url, resume_text"
        ).eq("id", persona_id).single().execute()
        
        persona = response.data
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        has_resume_url = bool(persona.get("resume_url"))
        has_summary = bool(persona.get("resume_text") and len(str(persona.get("resume_text", "")).strip()) > 50)
        
        return {
            "persona_id": persona_id,
            "has_resume_url": has_resume_url,
            "has_summary": has_summary,
            "needs_processing": has_resume_url and not has_summary,
            "summary_length": len(str(persona.get("resume_text", ""))) if has_summary else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - CONFIGURATION
# ============================================================================

@app.get("/api/models/status")
async def get_model_status():
    """Get available AI models"""
    return {
        "default_provider": DEFAULT_MODEL,
        "available_providers": {
            "gpt-4o-mini": True,
            "gpt-4o": True
        },
        "configured": OPENAI_API_KEY is not None
    }

@app.get("/api/response-styles")
async def get_response_styles():
    """Get available response styles"""
    return {
        "styles": {
            style_id: {
                "name": config["name"],
                "description": config["prompt"][:100] + "..."
            }
            for style_id, config in RESPONSE_STYLES.items()
        },
        "default": "concise"
    }

# ============================================================================
# NO HANDLER HERE! The handler is in api/index.py
# ============================================================================