import torch


class MaskNotAvailable(Exception):
    def __init__(self) -> None:
        super().__init__("Masks are not set by user. Please set the masks using set_current_masks method.")


class MaskShapeError(Exception):
    def __init__(self, shape: torch.Size) -> None:
        super().__init__(f"The masks should be a 4D tensor with shape (N, C, H, W). Got shape: {shape}")


class ModelNotSupportedError(Exception):
    def __init__(self, model_name: str, supported_models: str) -> None:
        super().__init__(
            f"Model {model_name} is not supported. Please choose one of the supported models: {supported_models}"
        )


class FormatNotSupportedError(Exception):
    def __init__(self, format_name: str, supported_formats: str) -> None:
        super().__init__(
            f"Format {format_name} is not supported. Please choose one of the supported formats: {supported_formats}"
        )


class ModeError(Exception):
    def __init__(self, mode: str) -> None:
        super().__init__(
            f"Mode {mode} is not supported. Please choose one of the supported modes: ['train', 'val', 'test']"
        )
