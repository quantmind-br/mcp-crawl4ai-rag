"""
LLM API client module for the Crawl4AI MCP server.

This module provides OpenAI-compatible API clients for chat and embeddings operations,
with support for fallback configurations and flexible endpoint configuration.
"""

import os
import logging
import openai


def get_chat_client():
    """
    Get a configured OpenAI client for chat/completion operations.

    Supports flexible configuration through environment variables:
    - CHAT_API_KEY: API key for chat model
    - CHAT_API_BASE: Base URL for chat API (defaults to OpenAI)

    Returns:
        openai.OpenAI: Configured OpenAI client for chat operations

    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration
    api_key = os.getenv("CHAT_API_KEY")
    base_url = os.getenv("CHAT_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for chat model. Please set CHAT_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using custom chat API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_embeddings_client():
    """
    Get a configured OpenAI client for embeddings operations.

    Supports flexible configuration through environment variables:
    - EMBEDDINGS_API_KEY: API key for embeddings
    - EMBEDDINGS_API_BASE: Base URL for embeddings API (defaults to OpenAI)

    Returns:
        openai.OpenAI: Configured OpenAI client for embeddings operations

    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for embeddings. Please set EMBEDDINGS_API_KEY"
        )

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_chat_fallback_client():
    """
    Get a configured OpenAI client for fallback chat/completion operations.

    Supports flexible configuration through environment variables with inheritance:
    - CHAT_FALLBACK_API_KEY: API key for fallback chat (inherits CHAT_API_KEY if not set)
    - CHAT_FALLBACK_API_BASE: Base URL for fallback chat API (inherits CHAT_API_BASE if not set)

    Returns:
        openai.OpenAI: Configured OpenAI client for fallback chat operations

    Raises:
        ValueError: If no API key is configured (primary or fallback)
    """
    # Get fallback configuration with inheritance
    api_key = os.getenv("CHAT_FALLBACK_API_KEY") or os.getenv("CHAT_API_KEY")
    base_url = os.getenv("CHAT_FALLBACK_API_BASE") or os.getenv("CHAT_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for fallback chat model. Please set CHAT_FALLBACK_API_KEY or CHAT_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using fallback chat API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint for fallback chat")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_embeddings_fallback_client():
    """
    Get a configured OpenAI client for fallback embeddings operations.

    Supports flexible configuration through environment variables with inheritance:
    - EMBEDDINGS_FALLBACK_API_KEY: API key for fallback embeddings (inherits EMBEDDINGS_API_KEY if not set)
    - EMBEDDINGS_FALLBACK_API_BASE: Base URL for fallback embeddings API (inherits EMBEDDINGS_API_BASE if not set)

    Returns:
        openai.OpenAI: Configured OpenAI client for fallback embeddings operations

    Raises:
        ValueError: If no API key is configured (primary or fallback)
    """
    # Get fallback configuration with inheritance
    api_key = os.getenv("EMBEDDINGS_FALLBACK_API_KEY") or os.getenv(
        "EMBEDDINGS_API_KEY"
    )
    base_url = os.getenv("EMBEDDINGS_FALLBACK_API_BASE") or os.getenv(
        "EMBEDDINGS_API_BASE"
    )

    if not api_key:
        raise ValueError(
            "No API key configured for fallback embeddings. Please set EMBEDDINGS_FALLBACK_API_KEY or EMBEDDINGS_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using fallback embeddings API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint for fallback embeddings")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_adaptive_chat_client(model_preference=None):
    """
    Get an adaptive OpenAI client that tries primary first, then fallback.

    Args:
        model_preference: Optional model name preference (uses environment fallback if None)

    Returns:
        tuple: (client, model_used, is_fallback)
            - client: Configured OpenAI client
            - model_used: The model name that will be used
            - is_fallback: Boolean indicating if fallback configuration is being used

    Raises:
        ValueError: If neither primary nor fallback configuration is available
    """
    # Determine model to use
    if model_preference:
        model_used = model_preference
        is_fallback = False
    else:
        # Try primary model first
        primary_model = os.getenv("CHAT_MODEL")
        if primary_model:
            model_used = primary_model
            is_fallback = False
        else:
            # Fall back to fallback model
            fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
            model_used = fallback_model
            is_fallback = True

    # Try to get appropriate client
    try:
        if not is_fallback and os.getenv("CHAT_API_KEY"):
            # Try primary client first
            client = get_chat_client()
            return (client, model_used, False)
        else:
            # Use fallback client
            client = get_chat_fallback_client()
            # If we're using fallback client, make sure we're using fallback model
            if not model_preference:
                fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
                model_used = fallback_model
            return (client, model_used, True)
    except ValueError as e:
        # If primary fails, try fallback
        if not is_fallback:
            try:
                client = get_chat_fallback_client()
                fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
                return (client, fallback_model, True)
            except ValueError:
                pass

        # Both failed
        raise ValueError(
            f"No valid API configuration available for chat model. "
            f"Please configure CHAT_API_KEY or CHAT_FALLBACK_API_KEY. Original error: {str(e)}"
        )


def validate_chat_config() -> bool:
    """
    Validate chat model configuration and provide helpful guidance.

    Returns:
        bool: True if configuration is valid, False otherwise

    Raises:
        ValueError: If critical configuration is missing
    """
    # Check for API key
    chat_api_key = os.getenv("CHAT_API_KEY")

    if not chat_api_key:
        raise ValueError(
            "No API key configured for chat model. Please set CHAT_API_KEY"
        )

    # Check for model configuration
    chat_model = os.getenv("CHAT_MODEL")

    if not chat_model:
        logging.warning(
            "No chat model specified. Please set CHAT_MODEL environment variable. "
            "Defaulting to configured fallback model."
        )

    # Log configuration being used
    effective_key_source = "CHAT_API_KEY"
    effective_model = chat_model or os.getenv("CHAT_FALLBACK_MODEL") or "default"
    base_url = os.getenv("CHAT_API_BASE", "default OpenAI")

    logging.debug(
        f"Chat configuration - Model: {effective_model}, Key source: {effective_key_source}, Base URL: {base_url}"
    )

    return True


def validate_chat_fallback_config() -> bool:
    """
    Validate chat fallback model configuration with inheritance support.

    Returns:
        bool: True if fallback configuration is valid (direct or inherited)

    Raises:
        ValueError: If no valid fallback configuration is available
    """
    # Check for fallback API key (direct or inherited)
    fallback_api_key = os.getenv("CHAT_FALLBACK_API_KEY") or os.getenv("CHAT_API_KEY")

    if not fallback_api_key:
        raise ValueError(
            "No API key configured for chat fallback. Please set CHAT_FALLBACK_API_KEY or CHAT_API_KEY"
        )

    # Validate base URL format if provided
    fallback_base_url = os.getenv("CHAT_FALLBACK_API_BASE") or os.getenv(
        "CHAT_API_BASE"
    )
    if fallback_base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(fallback_base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base URL format: {fallback_base_url}")
        except Exception as e:
            raise ValueError(f"Invalid fallback base URL configuration: {str(e)}")

    # Log effective configuration for debugging
    effective_key_source = (
        "CHAT_FALLBACK_API_KEY"
        if os.getenv("CHAT_FALLBACK_API_KEY")
        else "CHAT_API_KEY (inherited)"
    )
    effective_base_source = (
        "CHAT_FALLBACK_API_BASE"
        if os.getenv("CHAT_FALLBACK_API_BASE")
        else "CHAT_API_BASE (inherited)"
        if fallback_base_url
        else "default OpenAI"
    )
    fallback_model = os.getenv("CHAT_FALLBACK_MODEL", "gpt-4o-mini")

    logging.debug(
        f"Chat fallback configuration - Model: {fallback_model}, Key source: {effective_key_source}, Base URL source: {effective_base_source}"
    )

    return True


def validate_embeddings_fallback_config() -> bool:
    """
    Validate embeddings fallback model configuration with inheritance support.

    Returns:
        bool: True if fallback configuration is valid (direct or inherited)

    Raises:
        ValueError: If no valid fallback configuration is available
    """
    # Check for fallback API key (direct or inherited)
    fallback_api_key = os.getenv("EMBEDDINGS_FALLBACK_API_KEY") or os.getenv(
        "EMBEDDINGS_API_KEY"
    )

    if not fallback_api_key:
        raise ValueError(
            "No API key configured for embeddings fallback. Please set EMBEDDINGS_FALLBACK_API_KEY or EMBEDDINGS_API_KEY"
        )

    # Validate base URL format if provided
    fallback_base_url = os.getenv("EMBEDDINGS_FALLBACK_API_BASE") or os.getenv(
        "EMBEDDINGS_API_BASE"
    )
    if fallback_base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(fallback_base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base URL format: {fallback_base_url}")
        except Exception as e:
            raise ValueError(f"Invalid fallback base URL configuration: {str(e)}")

    # Log effective configuration for debugging
    effective_key_source = (
        "EMBEDDINGS_FALLBACK_API_KEY"
        if os.getenv("EMBEDDINGS_FALLBACK_API_KEY")
        else "EMBEDDINGS_API_KEY (inherited)"
    )
    effective_base_source = (
        "EMBEDDINGS_FALLBACK_API_BASE"
        if os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
        else "EMBEDDINGS_API_BASE (inherited)"
        if fallback_base_url
        else "default OpenAI"
    )
    fallback_model = os.getenv("EMBEDDINGS_FALLBACK_MODEL", "text-embedding-3-small")

    logging.debug(
        f"Embeddings fallback configuration - Model: {fallback_model}, Key source: {effective_key_source}, Base URL source: {effective_base_source}"
    )

    return True


def get_effective_fallback_config():
    """
    Get the effective fallback configuration with inheritance resolved.
    Useful for debugging and monitoring.

    Returns:
        dict: Configuration details showing what will actually be used
    """
    config = {
        "chat_fallback": {
            "model": os.getenv("CHAT_FALLBACK_MODEL", "gpt-4o-mini"),
            "api_key_source": "CHAT_FALLBACK_API_KEY"
            if os.getenv("CHAT_FALLBACK_API_KEY")
            else "CHAT_API_KEY (inherited)"
            if os.getenv("CHAT_API_KEY")
            else None,
            "base_url": os.getenv("CHAT_FALLBACK_API_BASE")
            or os.getenv("CHAT_API_BASE"),
            "base_url_source": "CHAT_FALLBACK_API_BASE"
            if os.getenv("CHAT_FALLBACK_API_BASE")
            else "CHAT_API_BASE (inherited)"
            if os.getenv("CHAT_API_BASE")
            else "default OpenAI",
        },
        "embeddings_fallback": {
            "model": os.getenv("EMBEDDINGS_FALLBACK_MODEL", "text-embedding-3-small"),
            "api_key_source": "EMBEDDINGS_FALLBACK_API_KEY"
            if os.getenv("EMBEDDINGS_FALLBACK_API_KEY")
            else "EMBEDDINGS_API_KEY (inherited)"
            if os.getenv("EMBEDDINGS_API_KEY")
            else None,
            "base_url": os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
            or os.getenv("EMBEDDINGS_API_BASE"),
            "base_url_source": "EMBEDDINGS_FALLBACK_API_BASE"
            if os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
            else "EMBEDDINGS_API_BASE (inherited)"
            if os.getenv("EMBEDDINGS_API_BASE")
            else "default OpenAI",
        },
    }

    return config
