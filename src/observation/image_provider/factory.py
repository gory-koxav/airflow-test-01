# -*- coding: utf-8 -*-
"""
Image Provider Factory

Strategy Pattern의 Context 역할을 하며,
문자열 타입에 따라 적절한 ImageProvider 인스턴스를 생성합니다.
"""

from typing import Dict, Any, Type

from .base import ImageProvider
from .onvif_provider import OnvifImageProvider
from .file_provider import FileImageProvider


class ImageProviderFactory:
    """
    Image Provider Factory (Strategy Pattern Context)

    사용 예시:
        provider = ImageProviderFactory.create("onvif", timeout=10)
        provider = ImageProviderFactory.create("file", base_path="/data/images")
    """

    _providers: Dict[str, Type[ImageProvider]] = {
        "onvif": OnvifImageProvider,
        "file": FileImageProvider,
    }

    @classmethod
    def create(cls, provider_type: str, **kwargs: Any) -> ImageProvider:
        """
        지정된 타입의 ImageProvider 인스턴스를 생성합니다.

        Args:
            provider_type: Provider 타입 ("onvif", "file")
            **kwargs: Provider 생성자에 전달할 인자

        Returns:
            ImageProvider: 생성된 Provider 인스턴스

        Raises:
            ValueError: 알 수 없는 provider_type인 경우
        """
        provider_class = cls._providers.get(provider_type.lower())
        if provider_class is None:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider type: {provider_type}. Available: {available}"
            )
        return provider_class(**kwargs)

    @classmethod
    def register(cls, name: str, provider_class: Type[ImageProvider]) -> None:
        """
        새로운 Provider 타입을 등록합니다.

        Args:
            name: Provider 타입 이름
            provider_class: ImageProvider를 상속받은 클래스

        Raises:
            TypeError: provider_class가 ImageProvider를 상속받지 않은 경우
        """
        if not issubclass(provider_class, ImageProvider):
            raise TypeError(
                f"{provider_class.__name__} must inherit from ImageProvider"
            )
        cls._providers[name.lower()] = provider_class

    @classmethod
    def available_providers(cls) -> list:
        """등록된 Provider 타입 목록 반환"""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider_type: str) -> bool:
        """특정 Provider 타입이 등록되어 있는지 확인"""
        return provider_type.lower() in cls._providers
