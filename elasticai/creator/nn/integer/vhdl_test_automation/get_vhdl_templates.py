def get_vhdl_templates(padding_len, layer_name):
    if padding_len == 0:
        suffix = "_not_padding"
    else:
        suffix = "_zero_padding"

    template_file_name = f"{layer_name}{suffix}.tpl.vhd"
    test_template_file_name = f"{layer_name}{suffix}_tb.tpl.vhd"

    return template_file_name, test_template_file_name
