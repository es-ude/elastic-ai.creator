# Hardware Function ID 

The `HW_FUNCTION_ID` uniquely identifies a hw design. It is set in the
skeleton hardware module (hence the name). The skeleton id is 16 byte
long. To ensure uniqueness the id is obtained by running blake2b on all
files except for the skeleton_pkg itself. This way equal hw designs will
always have an equal id, while being able to tell different designs
apart.

The algorithm is basically as follows

    file_digests := []
    FOR file in files
        file_digests.append(hash(file))
    END FOR
    file_digests := sort(file_hashes)
    skeleton_id := hash(file_digests)

:::
#### Hw Function ID in VHDL
:::

In VHDL we include the hardware function id as a constant from a package
called `skeleton_pkg`.

## HW accelerator meta description 

We need to know a few things about a hw accelerator to be able to use
it. This information should be provided through a meta file in toml
format.

The fields are

:::{list-table} title
:widths: 10 15 30
:header-rows: 1
*   - field name
    - optional/required
    - description
*   - `meta_version`
    - required
    - the version of the meta description file format
*   - `creator_version`
    - required
    - version of the creator tool, taht was used to generate this hw function
*   - `hw_accelerator_version`
    - required
    - the version of this specific accelerator, e.g., `"0.1.2"` this could
    change, e.g., if you overhaul your network topology
*   - `hw_fucntion_id`
    - required
    - the automatically generated id
*   - `name`
    - optional
    - used to identify the accelerator in a human readable way, e.g., `"convnet"`
*   - `description`
    - optional
    - a long description that may contain additional documentation
:::
