from typing import Any, Dict, List, Optional, Union
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi import HTTPException, status

class ErrorResponse(JSONResponse):
    """
    Standardized error response
    
    Args:
        detail: Detailed error description
        status_code: HTTP status code
    """
    def __init__(
        self,
        detail: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        content = {"error": True, "detail": detail}
        super().__init__(content=content, status_code=status_code)

class SuccessResponse(JSONResponse):
    """
    Standardized success response
    
    Args:
        data: Data to return
        message: Optional message
        status_code: HTTP status code
    """
    def __init__(
        self,
        data: Optional[Any] = None,
        message: Optional[str] = None,
        status_code: int = status.HTTP_200_OK,
    ):
        content = {"success": True}
        if data is not None:
            content["data"] = data
        if message is not None:
            content["message"] = message
        super().__init__(content=content, status_code=status_code)

class AudioStreamingResponse(StreamingResponse):
    """
    Audio streaming response
    
    Args:
        content: Audio data or audio data generator
        session_id: Session ID for tracking
        status_code: HTTP status code
        headers: Additional headers
    """
    def __init__(
        self,
        content: Any,
        session_id: str,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None
    ):
        if headers is None:
            headers = {}
        
        # Add session ID to headers
        headers["X-Session-ID"] = session_id
        
        super().__init__(
            content=content,
            status_code=status_code,
            media_type="audio/wav",
            headers=headers
        )