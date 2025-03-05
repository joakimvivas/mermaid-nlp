from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Mermaid Diagram Generator")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Request and response models for the API
class DiagramRequest(BaseModel):
    prompt: str = Field(
        ..., 
        description="Description of the diagram to generate",
        example="A customer places an order through the website. The system checks inventory. If available, process payment."
    )
    diagram_type: Optional[str] = Field(
        default="flowchart",
        description="Type of diagram (flowchart, sequenceDiagram, classDiagram, stateDiagram)",
        example="flowchart"
    )

class DiagramResponse(BaseModel):
    mermaid_syntax: str = Field(
        ...,
        description="Generated Mermaid diagram syntax",
        example="flowchart TD\n  A[Start] --> B[Process]\n  B --> C[End]"
    )

# Parser to clean the output and ensure valid Mermaid syntax
class MermaidOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove markdown blocks and quotes
        text = text.replace("```mermaid", "").replace("```", "").strip()
        text = text.replace('"', '').replace("'", '')
        valid_starts = ["graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram"]
        if not any(text.strip().startswith(start) for start in valid_starts):
            raise ValueError("Output must be valid Mermaid syntax")
        return text

# Define the prompts for the diagram generation chain
system_prompt = """You are an expert at generating Mermaid diagram syntax. Always respond ONLY with valid Mermaid syntax.
For flowcharts, use TD (top-down) direction and include clear node descriptions.
Use appropriate shapes for different node types:
- [] for process steps
- {{}} for decision points
- () for start/end points
Never include any explanations or markdown, only the Mermaid syntax."""

human_prompt = """Generate a {diagram_type} diagram for the following description: {prompt}"""

mermaid_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Function to generate a Mermaid diagram using the chain
def generate_mermaid_diagram(prompt: str, diagram_type: str) -> str:
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        mermaid_chain = mermaid_prompt | llm | MermaidOutputParser()
        result = mermaid_chain.invoke({"prompt": prompt, "diagram_type": diagram_type})
        return result
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        raise

# API endpoint to generate the diagram
@app.post("/api/generate-diagram", response_model=DiagramResponse, 
          summary="Generate Mermaid Diagram",
          description="Generates a Mermaid diagram based on the provided description")
async def api_generate_diagram(request: DiagramRequest):
    try:
        result = generate_mermaid_diagram(request.prompt, request.diagram_type)
        return DiagramResponse(mermaid_syntax=result)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Route to render the main page with the form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to process the form submission from the frontend
@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, prompt: str = Form(...), diagram_type: str = Form("flowchart")):
    try:
        result = generate_mermaid_diagram(prompt, diagram_type)
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "prompt": prompt, "diagram_type": diagram_type})
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
