# Court Document Processing System

A full-stack application with Python backend services and React frontend, featuring agent-based architecture and workflow management.

## 🏗️ Project Structure

```
├── be/                                 # Backend (Python)
│   ├── agents/                        # Agent-based architecture
│   │   ├── __init__.py
│   │   ├── action_matching_agent.py   # Action matching logic
│   │   ├── base_agent.py             # Base agent class
│   │   ├── customer_lookup_agent.py  # Customer lookup functionality
│   │   ├── execution_agent.py        # Task execution agent
│   │   ├── extraction_agent.py       # Data extraction agent
│   │   ├── logging_agent.py          # Logging functionality
│   │   ├── preprocessing_agent.py    # Data preprocessing
│   │   ├── review_router_agent.py    # Review routing logic
│   │   ├── state.py                  # State management
│   │   ├── validation_agent.py       # Data validation
│   │   └── workflow.py               # Workflow orchestration
│   ├── chroma_db/                    # ChromaDB vector database
│   ├── logs/                         # Application logs
│   ├── services/                     # Backend services
│   │   ├── action_executor.py        # Action execution service
│   ├── uploads/                      # File upload handling
│   ├── .env                          # Environment variables
│   ├── background.py                 # Background tasks
│   ├── config.py                     # Configuration settings
│   ├── database.py                   # Database configuration
│   ├── exceptions.py                 # Custom exceptions
│   ├── main.py                       # FastAPI main application
│   ├── metrics.py                    # Application metrics
│   ├── requirements.txt              # Python dependencies
│   └── vector_store.py               # Vector store operations
│
├── fe/                               # Frontend (React)
│   ├── public/                       # Static assets
│   │   ├── index.html               # Main HTML template
│   │   └── manifest.json            # PWA manifest
│   ├── src/                         # Source code
│   │   ├── components/              # React components
│   │   │   ├── FileUpload.jsx       # File upload component
│   │   │   ├── JobStatus.jsx        # Job status display
│   │   │   ├── Layout.jsx           # Layout component
│   │   │   ├── LoadingSpinner.jsx   # Loading indicator
│   │   │   ├── ReviewItem.jsx       # Review item component
│   │   │   └── StatsCard.jsx        # Statistics card
│   │   ├── pages/                   # Page components
│   │   │   ├── Dashboard.jsx        # Main dashboard
│   │   │   ├── Jobs.jsx             # Jobs management
│   │   │   ├── Review.jsx           # Review interface
│   │   │   ├── Settings.jsx         # Application settings
│   │   │   └── Upload.jsx           # File upload page
│   │   ├── services/                # API services
│   │   │   └── api.js               # API client
│   │   ├── utils/                   # Utility functions
│   │   │   └── helpers.js           # Helper functions
│   │   ├── App.jsx              # Main App component
│   │   ├── index.css            # Global styles
│   │   └── index.js             # Entry point
│   ├── package-lock.json            # Locked dependencies
│   ├── package.json                 # Node.js dependencies
│   ├── postcss.config.js            # PostCSS configuration
│   └── tailwind.config.js           # Tailwind CSS configuration
│
├── __init__.py                       # Python package initializer
├── Architecture Diagram.png         # System architecture diagram
├── customer_service.py              # Customer service module
├── llm_service.py                   # Large Language Model service
├── Notes.docx                       # Project documentation
├── pdf_service.py                   # PDF processing service
├── Sytem Workflow.docx              # Workflow documentation
├── Sytem Workflow.pdf               # Workflow documentation (PDF)
└── Workflow Diagram.png             # Workflow visualization
```

## 🚀 Features

- **Agent-Based Architecture**: Modular agent system for different functionalities
- **Workflow Management**: Automated workflow processing and routing
- **File Processing**: Support for CSV and PDF file uploads
- **Vector Database**: ChromaDB integration for semantic search
- **Real-time Updates**: Job status tracking and monitoring
- **Responsive UI**: Modern React frontend with Tailwind CSS
- **API Integration**: RESTful API with FastAPI backend

## 🛠️ Tech Stack

### Backend
- **Python 3.x** - Core backend language
- **FastAPI** - Modern web framework
- **ChromaDB** - Vector database for embeddings
- **SQLite/PostgreSQL** - Relational database
- **Pydantic** - Data validation

### Frontend
- **React** - Frontend framework
- **Tailwind CSS** - Utility-first CSS framework
- **JavaScript/JSX** - Frontend programming
- **PostCSS** - CSS processing

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn