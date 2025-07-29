# TruthLens - Advanced Fake News Detection System

<p align="center">
  <img src="https://raw.githubusercontent.com/SuryaKeyzz/Fake-News-Detection-pre-released-/main/assets/truthlens-logo.png" alt="TruthLens Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/username/Fake-News-Detection-pre-released-/releases">
    <img src="https://img.shields.io/github/v/release/username/Fake-News-Detection-pre-released-" alt="Version">
  </a>
  <a href="https://github.com/username/Fake-News-Detection-pre-released-/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/username/Fake-News-Detection-pre-released-" alt="License">
  </a>
  <a href="https://github.com/username/Fake-News-Detection-pre-released-/issues">
    <img src="https://img.shields.io/github/issues/username/Fake-News-Detection-pre-released-" alt="Issues">
  </a>
</p>

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Detailed Component Architecture](#-detailed-component-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Frontend Components](#-frontend-components)
- [Performance Optimization](#-performance-optimization)
- [Security Considerations](#-security-considerations)
- [Testing](#-testing)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

TruthLens is a comprehensive fake news detection system that combines multiple AI technologies to identify, analyze, and verify potentially misleading information on the web. At its core, the system leverages Spark LLM with Retrieval Augmented Generation (RAG) to perform context-aware fact-checking, supplemented by a suite of specialized analysis tools to evaluate content credibility, emotional manipulation, source reliability, and AI-generated content.

The system was designed to address the growing challenge of misinformation by providing:

1. **Evidence-based fact verification** through retrieving and analyzing relevant information
2. **Multi-dimensional trust assessment** that examines content, source, and contextual factors
3. **Transparency in reasoning** with detailed explanations of verdicts and confidence scoring
4. **User-friendly interface** that makes complex analysis accessible to everyday users

TruthLens can analyze both direct claims and full articles via URL, producing comprehensive reports with confidence scores calibrated across multiple factors.

## ✨ Features

### Core Analysis Capabilities

- **Claim & URL Analysis**: 
  - Process free-text claims entered directly by users
  - Extract and analyze full articles from URLs with specialized handling for major news sites (CNN, BBC, NYTimes, etc.)
  - Automated metadata extraction (author, publication date, title, domain)

- **RAG-Enhanced Fact Checking**: 
  - Web search integration for retrieving relevant evidence (using Google Search API)
  - Embedding-based similarity ranking to identify most relevant sources
  - FAISS vector indexing for efficient similarity search
  - Spark LLM analysis with structured chain-of-thought reasoning

- **Multi-pass Verification**: 
  - Initial fact-check assessment followed by secondary verification analysis
  - Calibrated confidence scores weighted by evidence quality
  - Two-stage reasoning process to reduce hallucination and improve accuracy
  - Support for "Partially True" verdicts with nuanced explanations

- **Named Entity Recognition**: 
  - Extraction of key entities (PERSON, ORG, GPE, DATE, PERCENT, MONEY, QUANTITY)
  - Entity verification through domain knowledge
  - Cross-referencing entities between claims and evidence
  - Entity coverage metrics for confidence calibration

### Content Analysis

- **Emotional Manipulation Detection**: 
  - Sentiment analysis using VADER for polarity scoring
  - Emotional keyword density analysis across multiple emotion categories
  - Sentence-level polarity distribution statistics
  - Detection of emotional manipulation techniques with quantified scoring

- **Propaganda Technique Detection**:
  - Identification of 8 common propaganda techniques:
    - Name calling
    - Bandwagon
    - Fear mongering
    - Appeal to authority
    - False dilemma
    - Straw man arguments
    - Ad hominem attacks
    - Whataboutism
  - Pattern-based detection with specific keyword triggers
  - Calculation of propaganda density scores

- **AI Content Detection**: 
  - Multi-factor detection using linguistic patterns, structural analysis, and language model assessment
  - Detection of common AI writing patterns (transition phrases, standardized structures)
  - Paragraph length variance and sentence structure analysis
  - Personal pronoun and contraction usage assessment
  - Content categorization (official news, educational, opinion, etc.)

- **Source Credibility Assessment**:
  - Domain reputation database with credibility scoring
  - Author credibility assessment
  - Publication date recency and relevance evaluation
  - Source transparency metrics
  - Trust Lens composite scoring integrating multiple credibility factors

- **Title-Content Contradiction Analysis**:
  - Detection of misleading headlines that don't match article content
  - Severity scoring for contradictions
  - Classification of contradiction types (slight mismatch to significant contradiction)

### User Experience

- **Interactive Dashboard**: 
  - Monthly fake news trend visualization
  - Category breakdown of misinformation
  - Source analysis statistics
  - Credibility factor metrics

- **Visual Result Presentation**: 
  - Color-coded verdict badges
  - Interactive confidence gauges and charts
  - Animated results reveal
  - Trust score 3D visualization using Three.js

- **Advanced UI Components**:
  - CredibilityHeatmap for intuitive trust visualization
  - Interactive AI detection panels with expandable details
  - Emotional analysis charts showing manipulation factors
  - Enhanced verdict assessment panels with confidence factors

- **User Assistance**:
  - Prompt quality analysis with real-time suggestions
  - Demo samples for educational purposes
  - Hover details for technical terms
  - Responsive design for all devices

- **Output Options**:
  - PDF report generation with comprehensive analysis
  - Social media sharing capabilities
  - Analysis history tracking
  - Clip-to-clipboard functionality

## 🔧 Tech Stack

### Backend

- **Python 3.8+**: Core programming language
- **FastAPI**: High-performance asynchronous API framework
  - Pydantic models for request/response validation
  - CORS middleware for cross-origin requests
  - Background tasks for non-blocking operations

- **Spark LLM**: Advanced language model integration
  - API-based integration with versioned models
  - Structured prompt engineering with chain-of-thought reasoning
  - Multiple evaluation passes for improved accuracy

- **Natural Language Processing**:
  - **spaCy**: Industrial-strength NLP with en_core_web_sm model
  - **NLTK**: Comprehensive NLP toolkit
    - VADER for sentiment analysis
    - Punkt for sentence tokenization
  - **SentenceTransformer**: Semantic embedding models
  - **Hugging Face Transformers**: Pre-trained language models

- **Vector Search**:
  - **FAISS**: Facebook AI Similarity Search for efficient vector operations
  - Cosine similarity metrics for relevance ranking
  - L2 distance calculations for vector matching

- **Web Technologies**:
  - **Requests**: HTTP library for API calls and web scraping
  - **Beautiful Soup 4**: HTML parsing for content extraction
  - **Trafilatura**: Clean web content extraction

- **Data Processing**:
  - **NumPy**: Numerical computing for vector operations
  - **Pandas**: Data manipulation and analysis
  - **Regular Expressions**: Pattern matching for text analysis

- **Optional Components**:
  - **Neo4j**: Graph database for knowledge representation
  - **doTenv**: Environment variable management

### Frontend

- **React 18+**: Component-based UI library
  - Functional components with hooks
  - Context API for state management
  - Custom hooks for reusable logic

- **Data Visualization**:
  - **Recharts**: Responsive chart components
    - LineChart for trend visualization
    - BarChart for comparative analysis
    - PieChart for distribution visualization
  - **Three.js**: 3D visualization library
    - Custom WebGL rendering for trust badges
    - Animated scenes with lighting effects

- **UI Components**:
  - **Tailwind CSS**: Utility-first styling
  - **Lucide React**: Comprehensive icon set
  - **Custom animations**: CSS and JS-based transitions

- **Advanced Features**:
  - **Canvas API**: For particle effects and water animation
  - **PDF generation**: Client-side PDF creation
  - **Clipboard API**: For sharing results
  - **AbortController**: For cancellable fetch requests

## 🏗 Detailed Component Architecture

TruthLens follows a modular architecture with specialized components:

### 1. Web Search Engine
```python
class WebSearchEngine:
```
- Retrieves relevant articles using Google Search API
- Handles API request formatting and response parsing
- Implements caching for efficient operation
- Features extraction for article content with fallback mechanisms

### 2. URL Handler
```python
class URLHandler:
```
- Validates and processes URL inputs
- Extracts article content with site-specific handling for major news sources
- Extracts metadata (title, author, publication date)
- Handles different HTML structures across news sites

### 3. Embedding Engine
```python
class EmbeddingEngine:
```
- Creates vector embeddings using SentenceTransformer
- Computes similarity between claim and evidence texts
- Filters content based on relevance thresholds
- Prepares data for FAISS indexing

### 4. FAISS Indexer
```python
class FAISSIndexer:
```
- Creates efficient similarity search indices
- Performs vector similarity search with L2 distance
- Retrieves most relevant documents for a query
- Converts distance metrics to similarity scores

### 5. Entity Extractor
```python
class EntityExtractor:
```
- Extracts named entities using spaCy
- Categorizes entities by type
- Filters for key entity types
- Provides entity lists for verification

### 6. SparkLLMFactChecker
```python
class SparkLLMFactChecker:
```
- Core fact-checking component using Spark LLM
- Implements structured prompting for factual assessment
- Processes evidence and generates reasoned verdicts
- Includes multi-pass verification and fallback mechanisms
- Calibrates confidence based on multiple factors

### 7. Sentiment Analyzer
```python
class SentimentAnalyzer:
```
- Analyzes sentiment using VADER
- Detects emotional keywords and manipulation
- Calculates sentence polarity statistics
- Identifies propaganda techniques
- Generates manipulation scores and explanations

### 8. Credibility Analyzer
```python
class CredibilityAnalyzer:
```
- Evaluates source credibility based on domain reputation
- Assesses author credibility
- Analyzes publication date relevance
- Generates overall credibility scores
- Calculates Trust Lens composite scores

### 9. AI Content Detector
```python
class AIContentDetector:
```
- Identifies AI-generated content using multiple signals
- Analyzes linguistic patterns typical of AI writing
- Evaluates structural features (paragraph length consistency, etc.)
- Extracts and analyzes linguistic traits
- Categorizes content type and purpose

### 10. Prompt Quality Analyzer
```python
class PromptQualityAnalyzer:
```
- Evaluates user prompts for clarity and specificity
- Suggests improvements for better analysis
- Generates enhanced prompt versions
- Provides quality scoring for prompts

### 11. Main System
```python
class FakeNewsDetectionSystem:
```
- Orchestrates all components
- Manages analysis workflow
- Handles error cases and fallbacks
- Produces final output with integrated results

## 📦 Installation

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm/yarn
- Google Search API credentials (for web search)
- Spark LLM API credentials
- Optional: Neo4j database (for knowledge graph features)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/username/Fake-News-Detection-pre-released-.git
cd Fake-News-Detection-pre-released-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys:
# - GOOGLE_API_KEY
# - GOOGLE_CX
# - SPARK_API_PASSWORD
# - Optional: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

### Docker Setup (Optional)

```bash
# Build backend image
docker build -t truthlens-backend -f Dockerfile.backend .

# Build frontend image
docker build -t truthlens-frontend -f Dockerfile.frontend .

# Run containers
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

```
# API Credentials
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_custom_search_engine_id
SPARK_API_PASSWORD=your_spark_api_password

# Optional: Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Server Configuration
PORT=8000
DEBUG=False
CACHE_TIMEOUT=3600
MAX_SEARCH_RESULTS=10
```

### Model Configuration

The system uses several models and configurations that can be adjusted:

- **Embedding Model**: `distilbert-base-nli-mean-tokens` is the default model in `EmbeddingEngine`
- **Similarity Threshold**: The default threshold is `0.70` in `filter_by_threshold`
- **SpaCy Model**: `en_core_web_sm` is used for entity extraction
- **Spark LLM Model**: `4.0Ultra` is the default model
- **Credibility Database**: Domain credibility scores in `credible_news_domains`

## 🚀 Usage

### Command Line Interface

```bash
# Basic usage
python Try_train.py --claim "COVID-19 vaccines contain microchips for tracking people."

# URL analysis
python Try_train.py --claim "https://example.com/article"

# Disable features
python Try_train.py --claim "Your claim here" --no-rag --no-kg

# Run as API server
python Try_train.py --api --host 0.0.0.0 --port 8000
```

### API Usage

```bash
# Analyze a claim
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"claim": "COVID-19 vaccines contain microchips for tracking people.", "use_rag": true, "use_kg": true}'

# Analyze a URL
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"claim": "https://example.com/article", "use_rag": true, "use_kg": true}'

# Analyze prompt quality
curl -X POST http://localhost:8000/analyze-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Is this claim true?"}'
```

### Web Interface

1. Navigate to `http://localhost:3000` in your browser
2. The interface has several tabs:
   - **Analysis Tool**: The main interface for analyzing claims
   - **Statistics Dashboard**: View fake news trends and statistics
   - **Fake News Examples**: Browse examples of debunked content
   - **Demo Samples**: Try pre-configured examples
   - **About**: Learn about the system's capabilities

3. To analyze a claim:
   - Enter text or URL in the input field
   - Toggle options as needed (web search, knowledge graph)
   - Click "Analyze Claim"
   - View results with detailed explanations and visualizations

4. Advanced features:
   - Click on any section of the results to see more details
   - Use the share feature to export or share analysis
   - Save results as PDF for offline reference
   - View your analysis history

## 📘 API Documentation

### Endpoints

#### `POST /analyze`

Analyzes a claim or URL for fake news detection.

**Request Body**:
```json
{
  "claim": "string (required) - The claim to analyze or a URL",
  "use_rag": "boolean (optional, default: true) - Whether to use web search for context",
  "use_kg": "boolean (optional, default: true) - Whether to use knowledge graph",
  "is_url": "boolean (optional) - Flag to indicate if claim is a URL"
}
```

**Response**:
```json
{
  "status": "string - success or error",
  "claim": "string - The analyzed claim",
  "is_url_input": "boolean - Whether input was a URL",
  "verdict": "string - True/False/Partially True/Unverified",
  "confidence": "integer - Confidence percentage",
  "explanation": "string - Explanation of verdict",
  "reasoning": "string - Detailed reasoning process",
  "entities": "object - Detected named entities",
  "emotional_manipulation": {
    "score": "float - Manipulation score",
    "level": "string - LOW/MODERATE/HIGH",
    "explanation": "string - Explanation of manipulation techniques",
    "details": "object - Detailed analysis"
  },
  "ai_detection": {
    "ai_score": "float - AI content probability",
    "ai_verdict": "string - Assessment of AI generation",
    "content_category": "string - Type of content",
    "reasoning": "string - Reasoning for AI assessment",
    "linguistic_traits": "object - Language pattern analysis"
  },
  "verification_analysis": "string - Multi-pass verification details",
  "source_metadata": "object - Source information for URL inputs",
  "credibility_assessment": "object - Source credibility analysis",
  "title_content_contradiction": "object - Title vs. content analysis",
  "trust_lens_score": "float - Composite trust score",
  "processing_time": "float - Analysis duration in seconds"
}
```

#### `POST /analyze-prompt`

Analyzes a user prompt and provides suggestions for improvement.

**Request Body**:
```json
{
  "prompt": "string (required) - The user prompt to analyze"
}
```

**Response**:
```json
{
  "is_good_prompt": "boolean - Whether prompt is well-formed",
  "quality_score": "float - Quality rating from 0-1",
  "is_url": "boolean - Whether prompt is a URL",
  "suggestions": "array - List of improvement suggestions",
  "improved_prompt": "string - Enhanced version of prompt"
}
```

#### `GET /health`

Health check endpoint.

**Response**:
```json
{
  "status": "string - healthy/unhealthy",
  "version": "string - API version",
  "timestamp": "string - Current datetime"
}
```

## 🖥️ Frontend Components

The frontend is built with React and features several specialized components:

### Key Components

#### `TruthLensApp`
The main container component that manages state and orchestrates all interactions.

#### `CredibilityHeatmap`
```jsx
<CredibilityHeatmap score={0.85} level="HIGH" />
```
Visualizes source credibility with an interactive heatmap display.

#### `InteractiveAIDetectionBadge`
```jsx
<InteractiveAIDetectionBadge aiDetection={result.ai_detection} />
```
Displays AI content detection results with expandable details.

#### `EnhancedVerdictAssessment`
```jsx
<EnhancedVerdictAssessment result={result} />
```
Presents the verdict with confidence visualization and detailed reasoning.

#### `EmotionalAnalysisChart`
```jsx
<EmotionalAnalysisChart sentimentAnalysis={result.emotional_manipulation} />
```
Charts emotional content and manipulation techniques.

#### `TrustBadge3D`
```jsx
<TrustBadge3D score={result.trust_lens_score} />
```
3D visualization of trust score using Three.js.

#### `PromptSuggestionPanel`
```jsx
<PromptSuggestionPanel
  prompt={claim}
  suggestions={promptSuggestions}
  onUseImproved={useImprovedPrompt}
  isVisible={showPromptSuggestions}
/>
```
Provides real-time suggestions for improving user prompts.

### Visual Effects

- `ParticleBackground`: Renders animated particle effects
- `WaterWaveEffect`: Creates animated wave backgrounds
- `ResultsRevealAnimation`: Animated reveal of analysis results

### State Management

The application uses React's useState and useEffect hooks to manage:
- User input (`claim`, `useRag`, `useKg`)
- Analysis results (`result`)
- UI state (`activeTab`, `isAnalyzing`, `showHistory`)
- Analysis suggestions (`promptSuggestions`)

### API Integration

The frontend interacts with the backend through several fetch requests:
- `handleAnalyze()`: Submits claims for analysis
- `analyzePrompt()`: Gets suggestions for prompt improvement
- API calls are debounced and use AbortController for cancellation

## 🚄 Performance Optimization

### Backend Optimizations

- **Result Caching**: The system caches results using the `result_cache` dictionary
- **Debounced Processing**: The prompt analyzer implements debouncing to reduce API calls
- **Parallel Processing**: Background tasks are used for non-blocking operations
- **Optimized Retrieval**: FAISS enables efficient vector similarity search
- **Fallback Mechanisms**: Each component includes fallbacks for when primary methods fail
- **Configurable Depth**: Analysis depth can be adjusted via parameters

### Frontend Optimizations

- **Component Memoization**: React.useCallback and React.useMemo for expensive operations
- **Debounced User Input**: Input handling is debounced to reduce unnecessary API calls
- **Conditional Rendering**: Components render only when necessary
- **Optimized Animations**: Hardware-accelerated animations with reduced frame rates
- **Visibility Detection**: Animations pause when tab is not visible
- **Image Optimization**: Placeholder images with proper dimensions
- **Incremental Loading**: Data and UI elements load incrementally

## 🔒 Security Considerations

- **Input Validation**: All API inputs are validated using Pydantic models
- **CORS Configuration**: Properly configured CORS middleware
- **Rate Limiting**: Recommended addition for production deployment
- **API Key Security**: Environment variables for sensitive credentials
- **Error Handling**: Sanitized error responses that don't leak implementation details
- **Content Sanitization**: User-provided content is sanitized before processing

## 🧪 Testing

### Backend Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run a specific test
pytest tests/unit/test_sentiment_analyzer.py
```

### Frontend Testing

```bash
# Run React component tests
npm test

# Run specific component test
npm test -- -t "CredibilityHeatmap"

# Run end-to-end tests
npm run test:e2e
```

## 🔮 Future Work

### Enhanced Analysis

- **Multilingual Support**: Extend to languages beyond English
- **Domain-Specific Models**: Specialized models for politics, science, health, etc.
- **Multimodal Analysis**: Image and video content analysis
- **Temporal Analysis**: Track how claims evolve over time
- **Citation Verification**: Direct verification of specific citations and references

### Technical Improvements

- **Streaming Responses**: Implement streaming for real-time analysis updates
- **Self-hosted Models**: Options for on-premise deployment of language models
- **Vector Database Integration**: Replace in-memory FAISS with persistent vector database
- **Enhanced Knowledge Graph**: More comprehensive entity relationship database
- **Improved URL Handling**: Support for more complex web page structures

### User Experience

- **Browser Extension**: One-click analysis from any webpage
- **Personalized Analysis**: User preferences for analysis depth and focus
- **Batch Processing**: Analyze multiple claims or URLs simultaneously
- **Social Features**: Community verification and discussion
- **Mobile App**: Native mobile experience

### Enterprise Features

- **Team Workspaces**: Collaborative fact-checking for teams
- **API Rate Limiting**: Tiered access controls
- **Custom Branding**: White-label solutions
- **Integration SDK**: Embed in third-party applications
- **Analytics Dashboard**: Usage statistics and trends

## 👥 Project Status

This repository is currently a private project and not open for public contributions. The codebase is maintained by the original author and selected collaborators.

If you're interested in the technology or have questions about the implementation, please contact the repository owner directly. While we're not accepting public pull requests at this time, we appreciate feedback and suggestions for future development.

### Development Standards

For internal collaborators:

- Follow the established code style (PEP 8 for Python, Prettier for JavaScript)
- Add tests for new functionality
- Update documentation alongside code changes
- Use descriptive commit messages that reference internal task IDs



<p align="center">
  Made with ❤️ for a more truthful internet
</p>