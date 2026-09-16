"""
SupportDNA Agent — Groq LLM Client
==================================
Encapsulates official Groq Python SDK communication with robust error handling,
timeout management, latency benchmarking, and token observability.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("supportdna.groq_client")

try:
    from groq import Groq, APIError, APITimeoutError, RateLimitError
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False
    Groq = None
    APIError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception


@dataclass
class GroqGenerationResult:
    text: str
    success: bool
    latency_ms: float
    model: str
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class GroqClient:
    """
    Dedicated client for Groq Chat Completions.
    Handles timeout, rate limiting, and network failures with clean observability.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None
    ):
        self.api_key = api_key.strip() if api_key is not None else os.environ.get("GROQ_API_KEY", "").strip()
        self.model = model.strip() if model is not None else os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        
        # Generation configuration
        try:
            self.timeout = float(timeout or os.environ.get("GROQ_TIMEOUT", "15.0"))
        except (ValueError, TypeError):
            self.timeout = 15.0

        try:
            self.default_temperature = float(default_temperature or os.environ.get("GROQ_TEMPERATURE", "0.2"))
        except (ValueError, TypeError):
            self.default_temperature = 0.2

        try:
            self.default_max_tokens = int(default_max_tokens or os.environ.get("GROQ_MAX_TOKENS", "512"))
        except (ValueError, TypeError):
            self.default_max_tokens = 512

        self.client = None
        if GROQ_SDK_AVAILABLE and self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = Groq(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq SDK client: {e}")
                self.client = None

    def is_available(self) -> bool:
        """Returns True if Groq SDK is installed and a non-placeholder API key is configured."""
        if not GROQ_SDK_AVAILABLE:
            return False
        if not self.api_key or self.api_key.startswith("your_") or len(self.api_key) < 10:
            return False
        return self.client is not None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GroqGenerationResult:
        """
        Send a chat completion request to Groq with latency and error tracking.
        """
        if not self.is_available():
            return GroqGenerationResult(
                text="",
                success=False,
                latency_ms=0.0,
                model=self.model,
                error="Groq client unavailable (GROQ_API_KEY not set or invalid)"
            )

        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            text = ""
            if response.choices and len(response.choices) > 0:
                text = (response.choices[0].message.content or "").strip()

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }

            if not text:
                return GroqGenerationResult(
                    text="",
                    success=False,
                    latency_ms=latency_ms,
                    model=self.model,
                    token_usage=usage,
                    error="Groq returned empty response"
                )

            return GroqGenerationResult(
                text=text,
                success=True,
                latency_ms=latency_ms,
                model=self.model,
                token_usage=usage,
                error=None
            )

        except APITimeoutError as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Groq API call timed out after {latency_ms}ms: {e}")
            return GroqGenerationResult(
                text="",
                success=False,
                latency_ms=latency_ms,
                model=self.model,
                error=f"Groq API call timed out: {e}"
            )
        except RateLimitError as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Groq rate limit encountered: {e}")
            return GroqGenerationResult(
                text="",
                success=False,
                latency_ms=latency_ms,
                model=self.model,
                error=f"Groq rate limit exceeded: {e}"
            )
        except APIError as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Groq API error: {e}")
            return GroqGenerationResult(
                text="",
                success=False,
                latency_ms=latency_ms,
                model=self.model,
                error=f"Groq API error: {e}"
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Unexpected error during Groq generation: {e}")
            return GroqGenerationResult(
                text="",
                success=False,
                latency_ms=latency_ms,
                model=self.model,
                error=f"Unexpected error: {e}"
            )
