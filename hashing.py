import hashlib

def get_md5_hash(v):
    """
    Returns the MD5 hash value for a given string.

    Args:
        v (str): String
    
    Returns:
        str: MD5 hash
    """
    return hashlib.md5(v.encode()).hexdigest()
