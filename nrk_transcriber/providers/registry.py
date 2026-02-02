"""
Provider registry for auto-detecting and instantiating providers.
"""

import logging
from typing import Optional, Type
from urllib.parse import urlparse

from .base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for media providers.

    Handles auto-detection of providers based on URLs and
    provides access to registered providers.
    """

    _providers: dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, provider_class: Type[BaseProvider]) -> Type[BaseProvider]:
        """
        Register a provider class.

        Can be used as a decorator:
            @ProviderRegistry.register
            class MyProvider(BaseProvider):
                ...

        Args:
            provider_class: The provider class to register

        Returns:
            The provider class (for decorator usage)
        """
        provider_id = provider_class.PROVIDER_ID
        if not provider_id:
            raise ValueError(
                f"Provider {provider_class.__name__} must define PROVIDER_ID"
            )

        logger.debug(f"Registering provider: {provider_id}")
        cls._providers[provider_id] = provider_class
        return provider_class

    @classmethod
    def detect_provider(cls, url: str) -> Optional[BaseProvider]:
        """
        Auto-detect the appropriate provider for a URL.

        Checks each registered provider's can_handle() method
        to find one that supports the URL.

        Args:
            url: The URL to check

        Returns:
            An instantiated provider, or None if no match
        """
        for provider_id, provider_class in cls._providers.items():
            try:
                if provider_class.can_handle(url):
                    logger.debug(f"URL matched provider: {provider_id}")
                    return provider_class()
            except Exception as e:
                logger.warning(
                    f"Error checking provider {provider_id}: {e}"
                )
                continue

        return None

    @classmethod
    def get_provider(cls, provider_id: str) -> BaseProvider:
        """
        Get a provider by ID.

        Args:
            provider_id: The provider identifier (e.g., "nrk", "bbc")

        Returns:
            An instantiated provider

        Raises:
            ValueError if provider not found
        """
        if provider_id not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_id}. "
                f"Available: {available}"
            )
        return cls._providers[provider_id]()

    @classmethod
    def list_providers(cls) -> list[dict]:
        """
        List all registered providers.

        Returns:
            List of provider info dicts
        """
        result = []
        for provider_id, provider_class in cls._providers.items():
            result.append({
                "id": provider_id,
                "name": provider_class.PROVIDER_NAME,
                "domains": provider_class.SUPPORTED_DOMAINS,
                "language": provider_class.DEFAULT_LANGUAGE,
            })
        return result

    @classmethod
    def get_provider_for_domain(cls, domain: str) -> Optional[BaseProvider]:
        """
        Get a provider that handles a specific domain.

        Args:
            domain: The domain to check (e.g., "radio.nrk.no")

        Returns:
            An instantiated provider, or None if no match
        """
        domain = domain.lower()
        for provider_class in cls._providers.values():
            for supported in provider_class.SUPPORTED_DOMAINS:
                if domain == supported or domain.endswith(f".{supported}"):
                    return provider_class()
        return None
