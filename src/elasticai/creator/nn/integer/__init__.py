import warnings

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The 'old_module' module is deprecated and will be removed in a future version. "
    "Use 'elasticai.creator.nn.linear' instead.",
    DeprecationWarning,
    stacklevel=2,  # Shows the warning at the caller's level
)
