from importlib import import_module

__all__ = [
    "basic",
    "basic_watermark",
    "finite_multi_draft_pfr",
    "mc",
    "mc_watermark",
    "multi_draft_pfr",
    "pfr",
    "pfr_mpfr_firstarrival",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
