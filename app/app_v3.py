"""
Pragati Backend v3.0 - Integrated with CrewAI Multi-Agent Validation System
Flask application using 109+ specialized AI agents for comprehensive idea validation.
"""

import os
import asyncio
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, Response
from flask_cors import CORS
import queue
import threading
from dotenv import load_dotenv
import logging

# Import the new CrewAI integration
from crew_ai_integration import validate_idea, get_evaluation_framework_info, get_system_health
from pdf_generator import ValidationReportGenerator
from pitch_deck_processor import PitchDeckProcessor
from database_manager import get_database_manager
from report_endpoints import register_report_endpoints

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

# Initialize PDF generator, pitch deck processor, and database manager
pdf_generator = ValidationReportGenerator()
pitch_deck_processor = PitchDeckProcessor()

# Initialize database manager
try:
    db_manager = get_database_manager()
    logger.info("✅ Database manager initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize database manager: {e}")
    db_manager = None

# Register report management endpoints
register_report_endpoints(app)

# Global queue for real-time agent messages
agent_message_queue = queue.Queue()
active_connections = set()

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_info = get_system_health()
        return jsonify({
            "status": "healthy",
            "message": "Pragati Backend v3.0 with CrewAI Multi-Agent System",
            "system_info": health_info
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "error": str(e)
        }), 500


@app.route('/api/validate-idea', methods=['POST'])
def validate_idea_endpoint():
    """
    Main idea validation endpoint using 109+ CrewAI agents
    
    Request body:
    {
        "user_id": "string",
        "title": "string",
        "idea_name": "string",
        "idea_concept": "string", 
        "custom_weights": {optional dict of cluster weights}
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'title', 'idea_name', 'idea_concept']
        missing_fields = [field for field in required_fields if not data or field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        user_id = data['user_id'].strip()
        title = data['title'].strip()
        idea_name = data['idea_name'].strip()
        idea_concept = data['idea_concept'].strip()
        custom_weights = data.get('custom_weights')
        
        if not all([user_id, title, idea_name, idea_concept]):
            return jsonify({
                "error": "All required fields must be non-empty"
            }), 400
        
        logger.info(f"Starting validation for idea: {idea_name} (User: {user_id})")
        
        # Broadcast validation start
        broadcast_agent_message("System", f"🚀 Starting validation for '{idea_name}'", "system")
        
        # Run validation using CrewAI multi-agent system
        try:
            result = validate_idea(idea_name, idea_concept, custom_weights)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return jsonify({
                "error": f"Validation failed: {str(e)}"
            }), 500
        
        # Save to MongoDB
        report_id = None
        if db_manager:
            try:
                report_id = db_manager.save_validation_report(
                    user_id=user_id,
                    title=title,
                    validation_result=result,
                    idea_name=idea_name,
                    idea_concept=idea_concept,
                    source_type="manual"
                )
                logger.info(f"✅ Saved report to MongoDB: {report_id}")
                broadcast_agent_message("System", f"💾 Report saved to database: {report_id}", "success")
            except Exception as e:
                logger.error(f"❌ Failed to save to MongoDB: {e}")
                # Continue without failing the validation
        
        # Broadcast validation completion  
        overall_score = result.get('overall_score', 3.0)
        broadcast_agent_message("System", f"✅ Validation completed! Overall score: {overall_score:.2f}/5.0", "success")
        
        # Add report ID to result
        if report_id:
            result['report_id'] = report_id
            result['saved_to_database'] = True
        else:
            result['saved_to_database'] = False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Validation endpoint error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route('/api/framework-info', methods=['GET'])
def framework_info():
    """Get information about the evaluation framework"""
    try:
        framework_info = get_evaluation_framework_info()
        return jsonify({
            "success": True,
            "data": framework_info
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to get framework info",
            "details": str(e)
        }), 500


@app.route('/api/system-info', methods=['GET'])
def system_info():
    """Get comprehensive system information"""
    try:
        from crew_ai_integration import get_pragati_validator
        
        validator = get_pragati_validator()
        system_info = validator.get_system_info()
        
        return jsonify({
            "success": True,
            "data": system_info
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to get system info",
            "details": str(e)
        }), 500


@app.route('/api/agents', methods=['GET'])
def get_agents_info():
    """Get information about all validation agents"""
    try:
        from crew_ai_integration import get_pragati_validator
        
        validator = get_pragati_validator()
        
        agents_info = {}
        for agent_id, agent in validator.agents.items():
            agents_info[agent_id] = {
                "cluster": agent.cluster,
                "parameter": agent.parameter,
                "sub_parameter": agent.sub_parameter,
                "weight": agent.weight,
                "dependencies": agent.dependencies
            }
        
        return jsonify({
            "success": True,
            "total_agents": len(agents_info),
            "agents": agents_info,
            "cluster_distribution": validator.agent_factory.get_agent_count_by_cluster()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Failed to get agents info",
            "details": str(e)
        }), 500


@app.route('/api/validate-pitch-deck', methods=['POST'])
def validate_pitch_deck():
    """
    Validate idea from uploaded pitch deck (PDF or PowerPoint)
    
    Request: multipart/form-data with:
    - 'pitch_deck' file
    - 'user_id' string
    - 'title' string
    Optional: 'custom_weights' as JSON string
    """
    try:
        # Check required form fields
        if 'user_id' not in request.form or 'title' not in request.form:
            return jsonify({
                "error": "Missing required fields: user_id and title"
            }), 400
        
        user_id = request.form['user_id'].strip()
        title = request.form['title'].strip()
        
        if not user_id or not title:
            return jsonify({
                "error": "user_id and title cannot be empty"
            }), 400
        
        # Check if file was uploaded
        if 'pitch_deck' not in request.files:
            return jsonify({
                "error": "No pitch deck file uploaded. Please upload a PDF or PowerPoint file."
            }), 400
        
        file = request.files['pitch_deck']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected"
            }), 400
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.ppt', '.pptx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type: {file_ext}. Allowed types: PDF, PPT, PPTX"
            }), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process pitch deck
            logger.info(f"Processing pitch deck: {file.filename} (User: {user_id})")
            broadcast_agent_message("System", f"📄 Processing pitch deck: {file.filename}", "system")
            
            extracted_info = pitch_deck_processor.process_pitch_deck(temp_path)
            
            logger.info(f"Extracted idea: {extracted_info['idea_name']}")
            broadcast_agent_message("System", f"✅ Extracted idea: {extracted_info['idea_name']}", "success")
            
            # Get custom weights if provided
            custom_weights = None
            if 'custom_weights' in request.form:
                import json
                custom_weights = json.loads(request.form['custom_weights'])
            
            # Run validation
            logger.info(f"Starting validation for: {extracted_info['idea_name']}")
            broadcast_agent_message("System", f"🚀 Starting validation with 109+ agents", "system")
            
            result = validate_idea(
                extracted_info['idea_name'],
                extracted_info['idea_concept'],
                custom_weights
            )
            
            # Save to MongoDB
            report_id = None
            if db_manager:
                try:
                    report_id = db_manager.save_validation_report(
                        user_id=user_id,
                        title=title,
                        validation_result=result,
                        idea_name=extracted_info['idea_name'],
                        idea_concept=extracted_info['idea_concept'],
                        source_type="pitch_deck"
                    )
                    logger.info(f"✅ Saved pitch deck report to MongoDB: {report_id}")
                    broadcast_agent_message("System", f"💾 Report saved to database: {report_id}", "success")
                except Exception as e:
                    logger.error(f"❌ Failed to save to MongoDB: {e}")
                    # Continue without failing the validation
            
            # Add extracted information to result
            result['extracted_from_pitch_deck'] = True
            result['original_filename'] = file.filename
            result['extracted_idea_name'] = extracted_info['idea_name']
            
            # Add report ID to result
            if report_id:
                result['report_id'] = report_id
                result['saved_to_database'] = True
            else:
                result['saved_to_database'] = False
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Pitch deck validation error: {str(e)}")
        return jsonify({
            "error": "Failed to process pitch deck",
            "details": str(e)
        }), 500


@app.route('/api/test-validation', methods=['GET'])
def test_validation():
    """Test endpoint for quick validation testing"""
    try:
        test_result = validate_idea(
            "Smart Agriculture IoT Platform",
            "An IoT-based platform that uses sensors and AI to help farmers monitor soil conditions, weather patterns, and crop health in real-time, with automated irrigation and fertilizer recommendations optimized for Indian farming conditions."
        )
        
        return jsonify({
            "success": True,
            "test_result": {
                "overall_score": test_result["overall_score"],
                "validation_outcome": test_result["validation_outcome"],
                "processing_time": test_result.get("processing_time", 0),
                "agents_consulted": test_result.get("api_calls_made", 0),
                "consensus_level": test_result.get("consensus_level", 0)
            },
            "message": "Test validation completed successfully"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Test validation failed",
            "details": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Serve the main web UI"""
    return send_from_directory('../static', 'index.html')

@app.route('/reports.html', methods=['GET'])
def reports_page():
    """Serve the reports viewer page"""
    return send_from_directory('../static', 'reports.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Pragati Backend v3.0",
        "description": "AI-powered idea validation using 109+ specialized CrewAI agents",
        "version": "3.0.0",
        "endpoints": {
            "POST /api/validate-idea": "Validate ideas using multi-agent system (JSON input)",
            "POST /api/validate-pitch-deck": "Validate ideas from PDF/PPT pitch deck upload",
            "GET /api/framework-info": "Get evaluation framework information",
            "GET /api/system-info": "Get system information",
            "GET /api/agents": "Get information about all validation agents",
            "GET /api/test-validation": "Test the validation system",
            "GET /api/reports/<user_id>": "Get all reports for a user (JSON)",
            "GET /api/report/<report_id>": "Get specific report by ID (JSON)",
            "GET /report/<report_id>": "Get report data for UI display (JSON)",
            "GET /": "Main validation interface (web UI)",
            "GET /reports.html": "Reports viewer page (web UI)",
            "GET /health": "Health check endpoint"
        },
        "features": [
            "109+ specialized AI validation agents",
            "Inter-agent collaboration and dependency resolution",
            "Comprehensive Indian market analysis",
            "Real-time consensus building",
            "Detailed HTML reporting",
            "Beautiful modern web interface"
        ]
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf_report():
    """Generate PDF report for validation result"""
    try:
        data = request.get_json()
        idea_name = data.get('idea_name', '')
        idea_concept = data.get('idea_concept', '')
        
        if not idea_name or not idea_concept:
            return jsonify({
                "error": "Both idea_name and idea_concept are required"
            }), 400
        
        # Run validation first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate_idea(idea_name, idea_concept))
        loop.close()
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.pdf"
        output_path = os.path.join('../static/reports', filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate PDF
        pdf_path = pdf_generator.generate_report(result, idea_name, idea_concept, output_path)
        
        return jsonify({
            "success": True,
            "pdf_url": f"/static/reports/{filename}",
            "filename": filename,
            "validation_result": {
                "overall_score": result.overall_score,
                "validation_outcome": result.validation_outcome.value,
                "total_agents_consulted": result.total_agents_consulted
            }
        })
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return jsonify({
            "error": f"PDF generation failed: {str(e)}"
        }), 500

@app.route('/api/download-pdf/<filename>')
def download_pdf(filename):
    """Download generated PDF report"""
    try:
        return send_file(f'../static/reports/{filename}', as_attachment=True)
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        return jsonify({
            "error": f"PDF download failed: {str(e)}"
        }), 404

@app.route('/api/agent-stream')
def agent_stream():
    """Server-Sent Events endpoint for real-time agent messages"""
    def event_stream():
        connection_id = id(threading.current_thread())
        active_connections.add(connection_id)
        
        try:
            while connection_id in active_connections:
                try:
                    # Get message from queue with timeout
                    message = agent_message_queue.get(timeout=1.0)
                    yield f"data: {message}\n\n"
                    agent_message_queue.task_done()
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    import json
                    heartbeat = json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})
                    yield f"data: {heartbeat}\n\n"
        except GeneratorExit:
            active_connections.discard(connection_id)
        finally:
            active_connections.discard(connection_id)
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})

@app.route('/api/test-message', methods=['POST'])
def test_message():
    """Test endpoint to send a message via SSE"""
    try:
        data = request.get_json()
        agent_name = data.get('agent', 'Test Agent')
        message = data.get('message', 'Test message')
        message_type = data.get('type', 'info')
        
        broadcast_agent_message(agent_name, message, message_type)
        
        return jsonify({
            "success": True,
            "message": "Test message sent"
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

def broadcast_agent_message(agent_name, message, message_type="info"):
    """Broadcast message to all connected clients"""
    import json
    
    # Clean the message to ensure valid JSON
    clean_message = str(message).replace('\n', ' ').replace('\r', ' ').strip()
    clean_agent = str(agent_name).replace('\n', ' ').replace('\r', ' ').strip()
    
    data = {
        'type': str(message_type),
        'agent': clean_agent,
        'message': clean_message,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        json_data = json.dumps(data, ensure_ascii=False)
        agent_message_queue.put(json_data, block=False)
        logger.debug(f"Broadcast message: {json_data}")
    except (queue.Full, TypeError, ValueError) as e:
        logger.warning(f"Failed to broadcast message: {e}")
        pass

# Make broadcast function globally available
import builtins
builtins.broadcast_agent_message = broadcast_agent_message


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)
    
    logger.info("Starting Pragati Backend v3.0 with CrewAI Multi-Agent System")
    
    # Initialize the validation system on startup
    try:
        from crew_ai_integration import get_pragati_validator
        validator = get_pragati_validator()
        logger.info(f"✅ Successfully initialized {len(validator.agents)} validation agents")
    except Exception as e:
        logger.error(f"Failed to initialize validation system: {e}")
        exit(1)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
