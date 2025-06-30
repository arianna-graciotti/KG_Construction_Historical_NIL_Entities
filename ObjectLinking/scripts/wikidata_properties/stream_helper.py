#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata JSON Stream Helper

This scripts reads a Wikidata JSON dump from stdin and produces a properly
formatted JSON stream starting from the opening '[' character and ensuring
proper JSON formatting. This helps address issues with ijson's parsing.
"""

import sys
import os

def stream_from_first_bracket():
    """
    Read from stdin and output to stdout, but skip everything until the first '['.
    """
    found_bracket = False
    buffer_size = 8192  # Use a reasonable buffer size
    
    while not found_bracket:
        # Read a chunk
        chunk = sys.stdin.buffer.read(buffer_size)
        if not chunk:  # End of file
            sys.stderr.write("Error: Reached end of file without finding '['\n")
            sys.exit(1)
        
        # Look for the opening bracket
        pos = chunk.find(b'[')
        if pos >= 0:
            # Found it! Write from this position onwards
            sys.stdout.buffer.write(chunk[pos:])
            found_bracket = True
        # Otherwise continue reading
    
    # Now stream the rest of the file
    while True:
        chunk = sys.stdin.buffer.read(buffer_size)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)

if __name__ == "__main__":
    stream_from_first_bracket()