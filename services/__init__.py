"""
Services module
Provides smart service selection based on available modules
"""

def get_ai_service_module():
    """
    Get AI service module - returns enterprise if available, otherwise base
    """
    try:
        from enterprise.services import ai_service_enterprise
        return ai_service_enterprise
    except (ImportError, ModuleNotFoundError):
        from services import ai_service
        return ai_service
