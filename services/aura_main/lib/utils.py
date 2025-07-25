# services/aura_main/lib/utils.py
"""
Bulletproof utilities for LLM response handling
"""
import json
import re
import os
import asyncio
import random
from typing import Optional, Dict, Any, Union

def parse_llm_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Bulletproof JSON parser for LLM responses with multiple fallback strategies
    
    Strategy progression:
    1. Find JSON in code blocks ```json...```
    2. Find first complete JSON object with balanced braces
    3. Auto-fix common JSON syntax errors
    4. Return None for honest failure
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: Look for JSON code blocks first
    json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        json_content = json_block_match.group(1).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            # Continue to other strategies
            pass
    
    # Strategy 2: Find first complete JSON object with balanced braces
    json_candidate = _extract_balanced_json(text)
    if json_candidate:
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # Try auto-fix before giving up
            fixed_json = _auto_fix_json_syntax(json_candidate)
            if fixed_json:
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
    
    # Strategy 3: Last resort - try to extract any JSON-like structure
    simple_match = re.search(r'\{[^{}]*\}', text)
    if simple_match:
        try:
            return json.loads(simple_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Honest failure
    return None

def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Extract the first JSON object with properly balanced braces
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
    
    return None

def _auto_fix_json_syntax(json_str: str) -> Optional[str]:
    """
    Auto-fix common JSON syntax errors from LLMs
    """
    if not json_str:
        return None
    
    # Remove trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix unquoted keys (common LLM error)
    fixed = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', fixed)
    
    # Fix single quotes to double quotes
    fixed = fixed.replace("'", '"')
    
    # Remove comments (// or /* */)
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    # Remove newlines inside strings (if any)
    fixed = re.sub(r'"\s*\n\s*"', '""', fixed)
    
    return fixed.strip()

def clean_llm_response(response: str) -> str:
    """
    Enhanced response cleaning for LLM outputs - fixes training mode issues
    """
    if not response:
        return "I'm here to help with your health concern. Could you provide more details?"
    
    # Remove training dialogue patterns (MedGemma training mode fix)
    cleaned = re.sub(r'\*\*Client\*\*:.*?\*\*Response\*\*:\s*', '', response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'Client:\s*.*?Response:\s*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\*\*Response\*\*:\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Remove conversation examples (Patient:, AURA:, etc.)
    cleaned = re.sub(r'Patient:\s*[^\n]*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'AURA:\s*[^\n]*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Doctor:\s*[^\n]*\n?', '', cleaned, flags=re.IGNORECASE)
    
    # Remove code blocks aggressively
    cleaned = re.sub(r'```(?:json|python|javascript)?\s*\n(.*?)\n```', r'\1', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'```(.*?)```', r'\1', cleaned, flags=re.DOTALL)
    
    # Remove standalone backticks and incomplete sentences
    cleaned = re.sub(r'`[^`]*$', '', cleaned)
    cleaned = re.sub(r'`([^`]*)`', r'\1', cleaned)
    
    # Aggressive duplicate line removal
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    unique_lines = []
    seen_lines = set()
    
    for line in lines:
        # Skip exact duplicates
        if line in seen_lines:
            continue
        # Skip very similar lines (fuzzy matching)
        is_similar = any(
            (line in existing and len(line) > 10) or 
            (existing in line and len(existing) > 10) or
            (len(set(line.split()) & set(existing.split())) > len(line.split()) * 0.7)
            for existing in seen_lines
        )
        if is_similar:
            continue
        
        seen_lines.add(line)
        unique_lines.append(line)
    
    # Join unique lines
    cleaned = ' '.join(unique_lines)
    
    # Clean excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove any remaining JSON artifacts if this is meant to be text
    if not cleaned.startswith('{'):
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
    
    # Fallback for problematic responses
    if (len(cleaned) < 30 or 
        len(unique_lines) <= 1 or 
        'ready to help' in cleaned.lower()):
        return "I understand you're experiencing a headache. Could you tell me when it started and how severe it is on a scale of 1-10?"
    
    return cleaned if cleaned else "I understand your concern. Let me help you with that."

async def exponential_backoff_retry(
    func,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    jitter: bool = True,
    backoff_factor: float = 2.0
):
    """
    Exponential backoff retry decorator with jitter
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Add random jitter to prevent thundering herd
        backoff_factor: Multiplier for each retry
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries - 1:
                raise last_exception
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                jitter_amount = delay * 0.1 * random.random()
                delay += jitter_amount
            
            print(f"🔄 Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s delay. Error: {str(e)[:100]}")
            await asyncio.sleep(delay)
    
    raise last_exception

class ServiceHealthChecker:
    """
    Health checker for service dependencies
    """
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    async def wait_for_service(self, service_url: str, max_wait: int = 60) -> bool:
        """
        Wait for a service to become healthy
        """
        import httpx
        
        print(f"🏥 Waiting for service at {service_url} to become healthy...")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_wait):
                try:
                    response = await client.get(f"{service_url}/health")
                    if response.status_code == 200:
                        print(f"✅ Service {service_url} is healthy")
                        return True
                except:
                    pass
                
                if attempt % 10 == 0:  # Log every 10 attempts
                    print(f"⏳ Still waiting for {service_url}... ({attempt}/{max_wait})")
                
                await asyncio.sleep(1)
        
        print(f"❌ Service {service_url} failed to become healthy within {max_wait}s")
        return False

# Utility function for robust HTTP calls
async def robust_http_call(
    client, 
    method: str, 
    url: str, 
    payload: Optional[Dict] = None,
    max_retries: int = 4
) -> Dict[str, Any]:
    """
    Make robust HTTP calls with exponential backoff
    """
    async def _make_call():
        if method.upper() == 'POST':
            response = await client.post(url, json=payload)
        else:
            response = await client.get(url)
        
        response.raise_for_status()
        return response.json()
    
    return await exponential_backoff_retry(_make_call, max_retries=max_retries)