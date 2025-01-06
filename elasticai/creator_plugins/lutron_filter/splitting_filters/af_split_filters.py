from lutron.ir.utils.filter_parameters import FilterParameters


def build_af_parameters() -> list[tuple[FilterParameters, FilterParameters]]:
    current_channels = 1
    params = []

    def filter(w, o, s=1):
        nonlocal current_channels
        params.append(
            FilterParameters(
                kernel_size=w, in_channels=current_channels, out_channels=o, stride=s
            )
        )
        current_channels = o

    filter(w=10, o=6, s=3)
    filter(w=6, o=5)
    filter(w=6, o=5)
    filter(w=6, o=6)
    return params
