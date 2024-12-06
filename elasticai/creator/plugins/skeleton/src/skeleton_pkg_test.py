import hashlib

from .skeleton_pkg import SkeletonIdHash, set_id_for_pkg


def hash():
    return hashlib.blake2b(digest_size=10)


def test_hashing_two_times_yields_same_digest():
    h1 = hash()
    h2 = hash()
    h1.update(b"hello")
    h2.update(b"hello")
    assert h1.digest() == h2.digest()


def test_hashing_in_different_order_yields_different_digests():
    h1 = hash()
    h2 = hash()
    content = [b"hello", b"world"]
    for n in content:
        h1.update(n)

    for n in reversed(content):
        h2.update(n)

    assert h1.digest() != h2.digest()


def test_hash_str_tuples_one_by_one():
    s1 = SkeletonIdHash()
    s2 = SkeletonIdHash()
    a = ("hello", "world")
    b = ("foo", "bar")
    s1.update(a)
    s1.update(b)

    s2.update(b)
    s2.update(a)

    assert s1.digest() == s2.digest()


def test_hashing_different_values():
    s1 = SkeletonIdHash()
    s2 = SkeletonIdHash()
    a = ("hell", "world")
    b = ("hello", "world")
    s1.update(a)
    s2.update(b)

    assert s1.digest() != s2.digest()


def test_can_set_id():
    code = """
    package skeleton_pkg is
        type skeleton_id_t is array (0 to 15) of std_logic_vector(7 downto 0);
        constant SKELETON_ID : skeleton_id_t := (others => x"00");
    end package;
    """.splitlines()
    code = tuple(set_id_for_pkg(code, id=["ff"] * 16))
    line = code[3].strip()
    _id = line.strip().split(":=")[1].strip().strip(";").strip(")").strip("(")
    assert _id == ", ".join(['x"FF"'] * 16)
