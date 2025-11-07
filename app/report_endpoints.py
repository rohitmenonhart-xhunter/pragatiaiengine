"""
Report Management Endpoints for Pragati AI Engine
Returns structured JSON data - UI will handle presentation
"""

import logging
from flask import jsonify, request, send_file, Response, stream_with_context, render_template, send_from_directory
from database_manager import get_database_manager
from pdf_report_system import generate_validation_report
from pdf_report_system.report_writer import AIReportWriter
from pdf_report_system.data_processor import AgentDataProcessor
from datetime import datetime
import json
import queue
import threading
from io import BytesIO
import os

logger = logging.getLogger(__name__)


def register_report_endpoints(app):
    """Register report management endpoints with Flask app"""
    
    @app.route('/api/reports/<user_id>', methods=['GET'])
    def get_user_reports(user_id):
        """Get all reports for a specific user"""
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            limit = request.args.get('limit', 10, type=int)
            reports = db_manager.get_user_reports(user_id, limit)
            
            return jsonify({
                "user_id": user_id,
                "reports": reports,
                "count": len(reports)
            })
            
        except Exception as e:
            logger.error(f"Failed to get user reports: {e}")
            return jsonify({
                "error": "Failed to retrieve reports",
                "details": str(e)
            }), 500

    @app.route('/api/report/<report_id>', methods=['GET'])
    def get_report_data(report_id):
        """Get specific report data by ID (full detailed analysis)"""
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            return jsonify(report)
            
        except Exception as e:
            logger.error(f"Failed to get report: {e}")
            return jsonify({
                "error": "Failed to retrieve report",
                "details": str(e)
            }), 500

    @app.route('/report/<report_id>', methods=['GET'])
    def get_report_for_display(report_id):
        """
        Serve the dedicated report view page
        Returns HTML page that renders report content from agent conversations
        """
        try:
            # Verify report exists
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Serve the HTML report page
            static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
            return send_from_directory(static_dir, 'report_view.html')
            
        except Exception as e:
            logger.error(f"Failed to serve report page: {e}")
            return jsonify({
                "error": "Failed to load report page",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/generate', methods=['GET'])
    def generate_ai_report(report_id):
        """
        Generate AI-written comprehensive report from agent conversations
        Uses gpt-4.1-mini to create detailed bullet-pointed report
        Caches the result in MongoDB - only generates on first request
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            # Check if cached AI report exists
            cached_report = db_manager.get_ai_report(report_id)
            if cached_report:
                logger.info(f"✅ Returning cached AI report for {report_id}")
                return jsonify({
                    "success": True,
                    "report_id": report_id,
                    "ai_report": cached_report["ai_report"],
                    "cached": True,
                    "generated_at": cached_report["generated_at"].isoformat() if cached_report.get("generated_at") else None
                })
            
            # No cached report - need to generate
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Extract agent conversations
            processor = AgentDataProcessor(report)
            processed_data = processor.process_complete_report_data()
            
            if not processed_data or not processed_data.get('all_conversations'):
                # Debug: Check what data is available
                has_evaluated_data = bool(report.get('evaluated_data'))
                has_raw_result = bool(report.get('raw_validation_result'))
                has_raw_evaluated = bool(report.get('raw_validation_result', {}).get('evaluated_data'))
                
                logger.warning(f"Report {report_id} - evaluated_data: {has_evaluated_data}, raw_result: {has_raw_result}, raw_evaluated: {has_raw_evaluated}")
                
                return jsonify({
                    "error": "No agent conversations found in this report",
                    "message": "This report does not contain agent conversation data. Please run a new validation to generate conversations.",
                    "debug": {
                        "has_evaluated_data": has_evaluated_data,
                        "has_raw_validation_result": has_raw_result,
                        "has_raw_evaluated_data": has_raw_evaluated
                    }
                }), 404
            
            # Generate AI-written report using gpt-4.1-mini
            logger.info(f"🔄 Generating new AI report for {report_id} using {len(processed_data['all_conversations'])} conversations")
            
            writer = AIReportWriter(progress_callback=None)
            ai_report = writer.write_comprehensive_report(
                processed_data['all_conversations'],
                processed_data['metadata']
            )
            
            # Save to MongoDB for caching
            save_success = db_manager.save_ai_report(report_id, ai_report)
            if save_success:
                logger.info(f"✅ AI report generated and cached for {report_id}")
            else:
                logger.warning(f"⚠️ AI report generated but failed to cache for {report_id}")
            
            return jsonify({
                "success": True,
                "report_id": report_id,
                "ai_report": ai_report,
                "conversations_used": len(processed_data['all_conversations']),
                "cached": False,
                "saved_to_db": save_success
            })
            
        except Exception as e:
            logger.error(f"Failed to generate AI report: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": "Failed to generate AI report",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/download', methods=['GET'])
    def download_report_pdf(report_id):
        """
        Download report as PDF with progress streaming
        Uses Server-Sent Events (SSE) to stream progress updates
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Create a queue for progress updates
            progress_queue = queue.Queue()
            pdf_result = {'buffer': None, 'error': None, 'filename': None}
            
            def progress_callback(message: str, progress: float):
                """Callback to send progress updates"""
                try:
                    progress_queue.put({
                        'message': message,
                        'progress': progress
                    })
                except Exception as e:
                    logger.error(f"Error sending progress: {e}")
            
            def generate_pdf():
                """Generate PDF in background thread"""
                try:
                    # Generate PDF with progress callback
                    pdf_buffer = generate_validation_report(report, progress_callback=progress_callback)
                    
                    # Create filename
                    title = report.get('title', 'report').replace(' ', '_')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"validation_report_{title}_{timestamp}.pdf"
                    
                    pdf_result['buffer'] = pdf_buffer
                    pdf_result['filename'] = filename
                    progress_queue.put({'type': 'complete'})
                except Exception as e:
                    logger.error(f"Failed to generate PDF: {e}")
                    pdf_result['error'] = str(e)
                    progress_queue.put({'type': 'error', 'error': str(e)})
            
            # Start PDF generation in background thread
            pdf_thread = threading.Thread(target=generate_pdf, daemon=True)
            pdf_thread.start()
            
            def generate():
                """Generator function for SSE streaming"""
                try:
                    while True:
                        try:
                            # Get progress update (with timeout)
                            item = progress_queue.get(timeout=1)
                            
                            if item.get('type') == 'complete':
                                # PDF generation complete, send final message
                                yield f"data: {json.dumps({'message': 'PDF ready!', 'progress': 100, 'complete': True})}\n\n"
                                break
                            elif item.get('type') == 'error':
                                # Error occurred
                                error_msg = item.get('error', 'Unknown error')
                                yield f"data: {json.dumps({'message': f'Error: {error_msg}', 'progress': 0, 'error': True})}\n\n"
                                break
                            else:
                                # Progress update
                                yield f"data: {json.dumps({'message': item.get('message', ''), 'progress': item.get('progress', 0)})}\n\n"
                                
                        except queue.Empty:
                            # Send keepalive
                            yield f"data: {json.dumps({'message': 'Processing...', 'progress': -1})}\n\n"
                            
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {json.dumps({'message': f'Error: {str(e)}', 'progress': 0, 'error': True})}\n\n"
            
            # Return SSE response
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to start PDF generation: {e}")
            return jsonify({
                "error": "Failed to start PDF generation",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/download-pdf-file', methods=['GET'])
    def download_pdf_file(report_id):
        """
        Actually download the generated PDF file
        Called after progress streaming is complete
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Generate PDF (without progress for direct download)
            pdf_buffer = generate_validation_report(report)
            
            # Create filename
            title = report.get('title', 'report').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"validation_report_{title}_{timestamp}.pdf"
            
            # Send PDF file
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return jsonify({
                "error": "Failed to generate PDF",
                "details": str(e)
            }), 500
