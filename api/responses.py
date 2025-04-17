from typing import Any, Dict, List, Optional, Union
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi import HTTPException, status

class ErrorResponse(JSONResponse):
    """
    Стандартизированный ответ с ошибкой
    
    Args:
        detail: Детальное описание ошибки
        status_code: HTTP-код статуса
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
    Стандартизированный успешный ответ
    
    Args:
        data: Данные для возврата
        message: Опциональное сообщение
        status_code: HTTP-код статуса
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
    Ответ с потоковой передачей аудио
    
    Args:
        content: Аудиоданные или генератор аудиоданных
        session_id: ID сессии для отслеживания
        status_code: HTTP-код статуса
        headers: Дополнительные заголовки
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
        
        # Добавляем ID сессии в заголовки
        headers["X-Session-ID"] = session_id
        
        super().__init__(
            content=content,
            status_code=status_code,
            media_type="audio/wav",
            headers=headers
        )