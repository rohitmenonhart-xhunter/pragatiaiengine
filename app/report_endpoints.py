"""
Report Management Endpoints for Pragati AI Engine
Returns structured JSON data - UI will handle presentation
"""

import logging
from flask import jsonify, request, send_file
from database_manager import get_database_manager
from report_pdf_generator import generate_report_pdf
from datetime import datetime

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
        Get report data for UI display
        Returns structured JSON data - UI will handle HTML rendering
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
            
            # Return structured data for UI to render
            return jsonify(report)
            
        except Exception as e:
            logger.error(f"Failed to get report: {e}")
            return jsonify({
                "error": "Failed to retrieve report",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/download', methods=['GET'])
    def download_report_pdf(report_id):
        """
        Download report as PDF
        Generates a detailed, professional PDF document
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
            
            # Generate PDF
            pdf_buffer = generate_report_pdf(report)
            
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
