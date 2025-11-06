# Pragati AI Engine

**AI-Powered Startup Idea Validation Platform with Psychometric Profiling**

This is a comprehensive startup validation platform that uses 109+ specialized AI agents to evaluate business ideas and provides personalized insights based on entrepreneur psychometric profiles.

-

## Core Features

### 1. Multi-Agent Idea Validation
- **109+ Specialized AI Agents** that evaluate ideas from different perspectives
- Agents organized into 7 clusters: Core Idea, Market Opportunity, Execution, Business Model, Team, Compliance, Risk & Strategy
- Real-time agent collaboration and consensus building
- Comprehensive scoring across 47+ parameters

### 2. Pitch Deck Analysis
- Upload PDF or PowerPoint pitch decks
- Automatic extraction of idea name and concept
- Full validation on extracted information
- No manual input needed

### 3. Psychometric Assessment System
- **Dynamic 20-question assessment** generated using GPT-4.1
- Evaluates 10 entrepreneurial dimensions:
  - Leadership & Vision
  - Risk Tolerance
  - Resilience & Adaptability
  - Innovation & Creativity
  - Decision Making
  - Emotional Intelligence
  - Persistence & Grit
  - Strategic Thinking
  - Communication Skills
  - Problem Solving

### 4. User Profile Creation
- **Automatic profile generation** after psychometric assessment
- Stores entrepreneur's strengths, weaknesses, fit score
- Tracks validation history
- Provides personalized context for future validations

### 5. Personalized Validation
- Validation system checks for user profile
- Adds **personalized insights** based on your traits:
  - Strengths to leverage for this idea
  - Areas you need to focus on
  - Fit between your profile and the idea
- Recommendations tailored to your personality

### 6. Report Management
- All validations saved to MongoDB
- Beautiful web UI to view reports
- Detailed analysis with scores, recommendations, next steps
- **PDF download** for any report
- Reports include pitch deck improvements if uploaded

### 7. Professional PDF Reports
- Multi-page detailed reports
- Color-coded scores and sections
- Executive summary
- Recommendations and action items
- Ready to share with stakeholders

---

## How It Works

### Standard Flow:

```
1. User provides idea → 
2. 109 AI agents validate → 
3. Report generated with scores & insights → 
4. Save to database → 
5. Download PDF or view in UI
```

### With Psychometric Profiling:

```
1. New user takes psychometric assessment (one-time) →
2. System creates entrepreneur profile →
3. User validates idea →
4. System retrieves profile →
5. Validation includes personalized insights →
6. Report shows how idea fits user's strengths/weaknesses →
7. History tracked in profile
```

---

## Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: OpenAI GPT-4.1, CrewAI framework
- **Database**: MongoDB
- **Document Processing**: python-pptx, pypdf, pdfplumber
- **PDF Generation**: ReportLab
- **Frontend**: Vanilla JavaScript, Modern CSS

---

## API Endpoints

### Idea Validation
```bash
POST /api/validate-idea
# Body: { user_id, title, idea_name, idea_concept }
# Returns: Complete validation report with 109+ agent insights

POST /api/validate-pitch-deck
# Form-data: { user_id, title, pitch_deck (file) }
# Returns: Extracted idea + validation report
```

### Psychometric Assessment
```bash
POST /api/psychometric/generate
# Body: { num_questions: 10-20, user_id }
# Returns: Dynamic question set

POST /api/psychometric/evaluate
# Body: { user_id, user_name, questions_data, responses }
# Returns: Complete evaluation + creates profile
```

### User Profiles
```bash
GET /api/profile/<user_id>
# Returns: Complete entrepreneur profile

GET /api/profile/<user_id>/validation-context
# Returns: Personalized validation context
```

### Reports
```bash
GET /api/reports/<user_id>
# Returns: All validation reports for user

GET /api/report/<report_id>
# Returns: Specific report details

GET /api/report/<report_id>/download
# Returns: PDF file download
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- MongoDB instance
- OpenAI API key
- Conda (recommended)

### Installation

1. **Clone the repo**
```bash
git clone https://github.com/rohitmenonhart-xhunter/pragatiaiengine.git
cd pragatiaiengine
```

2. **Create conda environment**
```bash
conda create -n pragatiaiengine python=3.11
conda activate pragatiaiengine
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create `.env` file:
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# MongoDB
MONGODB_URL=your_mongodb_connection_string
DATABASE_NAME=pragati_ai

# CrewAI Settings (non-interactive)
CREWAI_TRACING_ENABLED=false
CREWAI_DISABLE_TELEMETRY=true
CREWAI_PROMPT_EXECUTION_TRACES=false

# Server
PORT=5001
FLASK_ENV=development
```

5. **Run the application**
```bash
python app/app_v3.py
```

6. **Access the application**
- Main UI: http://localhost:5001
- Reports: http://localhost:5001/reports.html
- API Docs: http://localhost:5001/api

---

## Project Structure

```
pragatiaiengine/
├── app/
│   ├── app_v3.py                      # Main Flask application
│   ├── crew_ai_integration.py         # 109 agent system
│   ├── crew_ai_validation/            # Agent framework
│   │   ├── core.py                    # CrewAI orchestration
│   │   ├── agent_factory.py           # Agent creation
│   │   └── agents/                    # Agent definitions
│   ├── psychometric_evaluator.py      # Psychometric assessment
│   ├── psychometric_endpoints.py      # Assessment APIs
│   ├── user_profile_manager.py        # Profile management
│   ├── pitch_deck_processor.py        # PDF/PPT processing
│   ├── database_manager.py            # MongoDB operations
│   ├── report_endpoints.py            # Report APIs
│   └── report_pdf_generator.py        # PDF generation
├── static/
│   ├── index.html                     # Main validation UI
│   └── reports.html                   # Reports viewer
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables
└── README.md                          # This file
```

---

## Key Features Explained

### 1. Why 109 Agents?

Each agent represents a different stakeholder or evaluation perspective:
- **Investors** evaluate ROI, scalability, market size
- **Customers** assess value proposition, usability
- **Competitors** analyze differentiation, moats
- **Regulators** check compliance, legal issues
- **Technical Experts** review feasibility, architecture
- And many more...

They collaborate and debate, simulating real-world feedback you'd get from multiple sources.

### 2. How Psychometric Profiling Helps

Not every entrepreneur is suited for every type of startup. The system:
- Assesses your natural strengths (leadership, risk tolerance, etc.)
- Identifies areas you need to improve
- Matches your profile against the idea requirements
- Provides personalized recommendations

For example:
- High risk tolerance + innovative idea = Good fit, push forward
- Low resilience + high-stress industry = Warning, build support system
- Strong leadership + team-heavy execution = Leverage strength, hire smart

### 3. PDF Reports

The PDF reports include:
- **Title page** with overall score
- **Executive summary** with key findings
- **Detailed analysis** for each evaluation criterion
- **Recommendations** with priority levels
- **Next steps** with actionable items
- Professional formatting, ready to share

---

## Database Collections

The system uses these MongoDB collections:

1. **validation_reports** - All idea validation results
2. **psychometric_assessments** - Generated question sets
3. **psychometric_evaluations** - Completed assessments
4. **user_profiles** - Entrepreneur profiles with history

---

## Performance

- **Question Generation**: ~25 seconds (GPT-4.1)
- **Psychometric Evaluation**: ~11 seconds (GPT-4.1)
- **Idea Validation**: 30-90 seconds (109 agents)
- **PDF Generation**: 0.5-2 seconds
- **Report Retrieval**: <100ms

---

## Configuration

### Models Used:
- **Validation Agents**: GPT-4o
- **Psychometric System**: GPT-4.1 (gpt-4-turbo)
- **Pitch Deck Processing**: GPT-4o

### Timeouts:
- Psychometric: 60 seconds
- Validation: No timeout (long-running)
- API calls: 30 seconds each

### Limits:
- Questions per assessment: 10-20
- Max PDF size: Based on content
- Concurrent validations: Supported

---

## What Makes This Different

1. **Multi-Agent System**: Not just one AI, but 109 specialized agents that collaborate
2. **Psychometric Integration**: Validation considers the entrepreneur, not just the idea
3. **Personalized Insights**: Every recommendation tailored to your strengths/weaknesses
4. **Comprehensive**: Covers everything from market to compliance to team
5. **Actionable**: Not just scores, but specific next steps
6. **Professional**: PDF reports ready to share with investors
7. **Tracked History**: See your progress as an entrepreneur

---

## Use Cases

- **Solo Founders**: Validate ideas before investing time/money
- **Accelerators**: Screen applications, guide cohorts
- **Investors**: Quick due diligence on early-stage ideas
- **Educators**: Teach entrepreneurship with real feedback
- **Corporate Innovation**: Evaluate internal startup ideas



## Known Limitations

- Requires OpenAI API access (costs apply)
- Initial validation takes 30-90 seconds
- Best results with detailed idea descriptions
- Psychometric assessment requires honest responses
- MongoDB required for profiles and history

---


## Contact

Built by Rohit Menon
- GitHub: [@rohitmenonhart-xhunter](https://github.com/rohitmenonhart-xhunter)

---

## Acknowledgments

- OpenAI for GPT models
- CrewAI framework for multi-agent orchestration
- MongoDB for flexible data storage

---

**Last Updated**: November 6, 2025  
**Version**: 3.0.0  
**Status**: Production Ready

